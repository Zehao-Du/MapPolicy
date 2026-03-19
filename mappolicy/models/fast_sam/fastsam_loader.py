import pathlib
import sys
import importlib
from typing import Optional, Union
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image


models_path = pathlib.Path(__file__).resolve().parents[2]
fastsam_repo_path = models_path / "models" / "fast_sam" / "FastSAM"
if str(fastsam_repo_path) not in sys.path:
	sys.path.append(str(fastsam_repo_path))

fastsam_module = importlib.import_module("fastsam")
FastSAM = fastsam_module.FastSAM
FastSAMPrompt = fastsam_module.FastSAMPrompt


class FastSAM_Loader:
	def __init__(
		self,
		ckpt_path: Optional[Union[str, pathlib.Path]] = None,
		device: Optional[str] = None,
		imgsz: int = 1024,
		conf: float = 0.4,
		iou: float = 0.9,
		retina_masks: bool = True,
	):
		self.device = device
		self.imgsz = imgsz
		self.conf = conf
		self.iou = iou
		self.retina_masks = retina_masks
		self.model = None

		if ckpt_path is not None:
			self.load_ckpt(ckpt_path)

	def load_ckpt(self, ckpt_path: Union[str, pathlib.Path]):
		ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
		if not ckpt_path.exists():
			raise FileNotFoundError(f"FastSAM checkpoint not found: {ckpt_path}")

		self.model = FastSAM(str(ckpt_path))
		return self.model

	@staticmethod
	def _to_pil_rgb(rgb: Union[np.ndarray, Image.Image, torch.Tensor]) -> Image.Image:
		if isinstance(rgb, Image.Image):
			return rgb.convert("RGB")

		if isinstance(rgb, torch.Tensor):
			rgb = rgb.detach().cpu().numpy()

		if not isinstance(rgb, np.ndarray):
			raise TypeError("rgb must be np.ndarray, PIL.Image, or torch.Tensor")

		if rgb.ndim != 3:
			raise ValueError(f"Expected rgb with 3 dims [H,W,C] or [C,H,W], got shape {rgb.shape}")

		if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
			rgb = np.transpose(rgb, (1, 2, 0))

		if rgb.shape[-1] != 3:
			raise ValueError(f"Expected 3 channels, got shape {rgb.shape}")

		if np.issubdtype(rgb.dtype, np.floating):
			rgb = np.clip(rgb, 0.0, 1.0) * 255.0

		rgb = rgb.astype(np.uint8)
		return Image.fromarray(rgb, mode="RGB")

	@staticmethod
	def _ensure_bool_mask(mask: Union[np.ndarray, torch.Tensor], h: int, w: int) -> np.ndarray:
		if isinstance(mask, torch.Tensor):
			mask = mask.detach().cpu().numpy()
		mask = np.asarray(mask)
		if mask.shape != (h, w):
			mask = mask.reshape(h, w)
		return mask > 0

	@staticmethod
	def _adjust_idx_for_filter(local_idx: int, filter_id: list) -> int:
		if len(filter_id) == 0:
			return int(local_idx)
		return int(local_idx) + int(sum(np.array(filter_id) <= int(local_idx)))

	def _ensure_clip_module(self):
		prompt_module = importlib.import_module("fastsam.prompt")
		if not hasattr(prompt_module, "clip"):
			try:
				import clip
				prompt_module.clip = clip
			except Exception as exc:
				raise ImportError(
					"CLIP is required for text prompt. Install with: "
					"pip install git+https://github.com/openai/CLIP.git"
				) from exc
		return prompt_module.clip

	def _build_prompt_process(self, rgb: Union[np.ndarray, Image.Image, torch.Tensor]):
		if self.model is None:
			raise RuntimeError("FastSAM model is not loaded. Call load_ckpt(...) first.")

		image = self._to_pil_rgb(rgb)
		everything_results = self.model(
			image,
			device=self.device,
			retina_masks=self.retina_masks,
			imgsz=self.imgsz,
			conf=self.conf,
			iou=self.iou,
		)
		prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
		return image, prompt_process

	def _text_prompt_candidates_from_process(
		self,
		prompt_process: "FastSAMPrompt",
		text_prompt: str,
		clip_model,
		preprocess,
		topk: int = 5,
	):
		if prompt_process.results is None:
			return []

		format_results = prompt_process._format_results(prompt_process.results[0], 0)
		cropped_boxes, _, not_crop, filter_id, annotations = prompt_process._crop_image(format_results)
		if len(cropped_boxes) == 0:
			return []

		scores = prompt_process.retrieve(clip_model, preprocess, cropped_boxes, text_prompt, device=self.device)
		if isinstance(scores, torch.Tensor):
			order = torch.argsort(scores, descending=True).detach().cpu().numpy()
			scores_np = scores.detach().cpu().numpy()
		else:
			scores_np = np.asarray(scores)
			order = np.argsort(scores_np)[::-1]

		if isinstance(prompt_process.img, np.ndarray):
			h, w = prompt_process.img.shape[0], prompt_process.img.shape[1]
		else:
			h, w = prompt_process.img.height, prompt_process.img.width
		candidates = []
		for local_idx in order[:max(1, int(topk))]:
			global_idx = self._adjust_idx_for_filter(int(local_idx), filter_id)
			if global_idx >= len(annotations):
				continue
			mask = annotations[global_idx]["segmentation"]
			mask = self._ensure_bool_mask(mask, h, w)
			if mask.sum() == 0:
				continue
			candidates.append({"mask": mask, "score": float(scores_np[local_idx])})

		return candidates

	def segment_multi_text_prompts(
		self,
		rgb: Union[np.ndarray, Image.Image, torch.Tensor],
		text_prompts,
		topk: int = 8,
		overlap_penalty: float = 0.8,
		min_pixels: int = 50,
	):
		if text_prompts is None or len(text_prompts) == 0:
			return OrderedDict()

		image, prompt_process = self._build_prompt_process(rgb)
		clip_module = self._ensure_clip_module()
		clip_model, preprocess = clip_module.load("ViT-B/32", device=self.device)
		image_np = np.asarray(image, dtype=np.uint8)

		h, w = image.height, image.width
		used = np.zeros((h, w), dtype=bool)
		result = OrderedDict()

		for prompt in text_prompts:
			prompt_l = str(prompt).lower()
			cands = self._text_prompt_candidates_from_process(
				prompt_process=prompt_process,
				text_prompt=prompt,
				clip_model=clip_model,
				preprocess=preprocess,
				topk=topk,
			)

			if len(cands) == 0:
				result[prompt] = np.zeros((h, w), dtype=bool)
				continue

			best_mask = cands[0]["mask"]
			best_adjusted = -1e9

			for rank, item in enumerate(cands):
				mask = item["mask"]
				score = float(item["score"])
				area = int(mask.sum())
				area_ratio = area / max(1, h * w)
				inter = int(np.logical_and(mask, used).sum())
				overlap_ratio = inter / max(1, area)
				new_pixels = area - inter

				adjusted = score - overlap_penalty * overlap_ratio - 0.02 * rank
				if new_pixels < min_pixels:
					adjusted -= 1.0

				# 颜色先验：帮助 red/green/blue 等提示词区分目标
				pixels = image_np[mask]
				if pixels.size > 0:
					mean_r, mean_g, mean_b = pixels.mean(axis=0)
					if "red" in prompt_l:
						adjusted += 0.4 * ((mean_r - mean_g) / 255.0)
					if "green" in prompt_l:
						adjusted += 0.4 * ((mean_g - max(mean_r, mean_b)) / 255.0)
					if "blue" in prompt_l:
						adjusted += 0.4 * ((mean_b - mean_r) / 255.0)

				# 尺度先验：对小物体词汇抑制大面积掩码
				if any(tok in prompt_l for tok in ["cube", "peg", "puck", "ball", "charger", "tool"]):
					adjusted -= 0.8 * max(0.0, area_ratio - 0.08)

				# 桌面词汇不应过小
				if any(tok in prompt_l for tok in ["desk", "table", "tabletop"]) and area_ratio < 0.03:
					adjusted -= 0.5

				if adjusted > best_adjusted:
					best_adjusted = adjusted
					best_mask = mask

			result[prompt] = best_mask
			used = np.logical_or(used, best_mask)

		return result

	def segment_by_text_prompt(
		self,
		rgb: Union[np.ndarray, Image.Image, torch.Tensor],
		text_prompt: str,
	) -> np.ndarray:
		"""
		输入 RGB 和文本提示，输出二值分割掩码（H, W）, dtype=bool。
		"""
		if not text_prompt or not isinstance(text_prompt, str):
			raise ValueError("text_prompt must be a non-empty string")
		masks = self.segment_multi_text_prompts(rgb, [text_prompt], topk=8)
		return masks[text_prompt]


def main():
	# 固定输入
	ckpt_path = "/data2/zehao/MapPolicy/mappolicy/models/fast_sam/FastSAM/weights/FastSAM-x.pt"
	image_path = "/data2/zehao/MapPolicy/mappolicy/models/fast_sam/FastSAM/images/cat.jpg"
	text_prompt = "cat"
	output_dir = pathlib.Path("/data2/zehao/MapPolicy/mappolicy/models/fast_sam/test_output")
	mask_output_path = output_dir / "cat_mask.png"
	masked_rgb_output_path = output_dir / "cat_masked_rgb.png"

	device = "cuda:0" if torch.cuda.is_available() else None
	loader = FastSAM_Loader(ckpt_path=ckpt_path, device=device)
	image = Image.open(image_path).convert("RGB")
	mask = loader.segment_by_text_prompt(image, text_prompt)

	print(f"[FastSAM_Loader] mask shape: {mask.shape}, dtype: {mask.dtype}, foreground pixels: {int(mask.sum())}")

	output_dir.mkdir(parents=True, exist_ok=True)

	# 1) 保存二值掩码
	Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_output_path)
	print(f"[FastSAM_Loader] mask saved to: {mask_output_path}")

	# 2) 保存掩码后的 RGB
	image_np = np.array(image, dtype=np.uint8)
	masked_rgb = np.zeros_like(image_np)
	masked_rgb[mask] = image_np[mask]
	Image.fromarray(masked_rgb).save(masked_rgb_output_path)
	print(f"[FastSAM_Loader] masked rgb saved to: {masked_rgb_output_path}")


if __name__ == "__main__":
	main()