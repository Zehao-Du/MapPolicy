import pathlib
import sys
import importlib
from typing import Optional, Union

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

	def segment_by_text_prompt(
		self,
		rgb: Union[np.ndarray, Image.Image, torch.Tensor],
		text_prompt: str,
	) -> np.ndarray:
		"""
		输入 RGB 和文本提示，输出二值分割掩码（H, W）, dtype=bool。
		"""
		if self.model is None:
			raise RuntimeError("FastSAM model is not loaded. Call load_ckpt(...) first.")
		if not text_prompt or not isinstance(text_prompt, str):
			raise ValueError("text_prompt must be a non-empty string")

		image = self._to_pil_rgb(rgb)

		everything_results = self.model(
			image,
			device=self.device,
			retina_masks=self.retina_masks,
			imgsz=self.imgsz,
			conf=self.conf,
			iou=self.iou,
		)

		# FastSAM 上游实现里 text_prompt() 依赖全局 clip 符号，这里确保其可用。
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

		prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
		ann = prompt_process.text_prompt(text=text_prompt)

		if ann is None or len(ann) == 0:
			return np.zeros((image.height, image.width), dtype=bool)

		mask = ann[0]
		if isinstance(mask, torch.Tensor):
			mask = mask.detach().cpu().numpy()

		mask = np.asarray(mask) > 0
		return mask


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