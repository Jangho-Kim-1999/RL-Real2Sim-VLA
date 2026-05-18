from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_NAVILA_MODEL = "a8cheng/navila-llama3-8b-8f"
DEFAULT_TARGET_NAMES = ("red object", "green object", "blue object")


def command_to_raw_action(command: torch.Tensor, velocity_range: torch.Tensor) -> torch.Tensor:
    """Map a desired body velocity to the raw high-level action expected by VelocityCommandAction."""
    normalized = torch.clamp(command / velocity_range, -0.95, 0.95)
    return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))


def _clean_number(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _first_match(patterns: list[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match
    return None


@dataclass
class ParsedPrimitive:
    velocity_command: np.ndarray
    hold_steps: int
    kind: str


class NaVILAActionParser:
    """Convert NaVILA's natural-language navigation action into a velocity primitive."""

    _TURN_PATTERNS = [
        r"\b(turn|rotate|rotation|yaw)\s+(?:to\s+the\s+)?(?P<dir>left|right|clockwise|counterclockwise)"
        r"(?:\s+by)?[^0-9+\-.]{0,24}(?P<value>[+\-]?\d+(?:\.\d+)?)\s*(?P<unit>degrees?|deg|radians?|rad|°)?",
        r"(?P<value>[+\-]?\d+(?:\.\d+)?)\s*(?P<unit>degrees?|deg|radians?|rad|°)"
        r"\s+(?:to\s+the\s+)?(?P<dir>left|right|clockwise|counterclockwise)",
    ]
    _MOVE_PATTERNS = [
        r"\b(move|go|walk|proceed|continue)\s+(?P<dir>forward|straight|ahead|backward|back)"
        r"[^0-9+\-.]{0,24}(?P<value>[+\-]?\d+(?:\.\d+)?)?\s*(?P<unit>meters?|metres?|m|centimeters?|centimetres?|cm)?",
        r"\b(?P<value>[+\-]?\d+(?:\.\d+)?)\s*(?P<unit>meters?|metres?|m|centimeters?|centimetres?|cm)"
        r"\s+(?P<dir>forward|straight|ahead|backward|back)",
    ]

    def __init__(
        self,
        policy_dt: float,
        forward_speed: float = 0.5,
        turn_speed: float = 0.8,
        default_forward_distance: float = 0.25,
        default_turn_degrees: float = 30.0,
        stop_hold_steps: int = 1,
        max_hold_steps: int = 60,
    ) -> None:
        self.policy_dt = max(float(policy_dt), 1e-6)
        self.forward_speed = abs(float(forward_speed))
        self.turn_speed = abs(float(turn_speed))
        self.default_forward_distance = abs(float(default_forward_distance))
        self.default_turn_degrees = abs(float(default_turn_degrees))
        self.stop_hold_steps = max(1, int(stop_hold_steps))
        self.max_hold_steps = max(1, int(max_hold_steps))

    def parse(self, text: str) -> ParsedPrimitive:
        lower = text.lower()
        if self._is_stop(lower):
            return ParsedPrimitive(np.zeros(3, dtype=np.float32), self.stop_hold_steps, "stop")

        turn_match = _first_match(self._TURN_PATTERNS, lower)
        move_match = _first_match(self._MOVE_PATTERNS, lower)
        if turn_match is not None and (move_match is None or turn_match.start() <= move_match.start()):
            return self._parse_turn(turn_match)
        if move_match is not None:
            return self._parse_move(move_match)

        if re.search(r"\b(left|counterclockwise)\b", lower):
            return self._turn(math.radians(self.default_turn_degrees), "turn_left")
        if re.search(r"\b(right|clockwise)\b", lower):
            return self._turn(-math.radians(self.default_turn_degrees), "turn_right")
        if re.search(r"\b(forward|straight|ahead)\b", lower):
            return self._move(self.default_forward_distance, "move_forward")

        return ParsedPrimitive(np.zeros(3, dtype=np.float32), self.stop_hold_steps, "unparsed")

    @staticmethod
    def _is_stop(text: str) -> bool:
        stop_words = ("stop", "completed", "complete", "arrived", "reached", "finished", "done")
        return any(re.search(rf"\b{word}\b", text) for word in stop_words)

    def _parse_turn(self, match: re.Match[str]) -> ParsedPrimitive:
        direction = (match.groupdict().get("dir") or "").lower()
        value = _clean_number(match.groupdict().get("value") or "")
        unit = (match.groupdict().get("unit") or "degree").lower()
        angle = math.radians(self.default_turn_degrees) if value is None else float(value)
        if unit.startswith("deg") or unit.startswith("degree") or unit == "°":
            angle = math.radians(angle)
        if direction in {"right", "clockwise"}:
            angle = -abs(angle)
        elif direction in {"left", "counterclockwise"}:
            angle = abs(angle)
        return self._turn(angle, "turn_left" if angle >= 0.0 else "turn_right")

    def _parse_move(self, match: re.Match[str]) -> ParsedPrimitive:
        direction = (match.groupdict().get("dir") or "forward").lower()
        value = _clean_number(match.groupdict().get("value") or "")
        unit = (match.groupdict().get("unit") or "").lower()
        distance = self.default_forward_distance if value is None else abs(float(value))
        if unit.startswith("centimeter") or unit.startswith("centimetre") or unit == "cm":
            distance *= 0.01
        elif unit == "" and value is not None and distance > 5.0:
            distance *= 0.01
        if direction in {"backward", "back"}:
            distance = -distance
        return self._move(distance, "move_forward" if distance >= 0.0 else "move_backward")

    def _turn(self, angle_rad: float, kind: str) -> ParsedPrimitive:
        angular_velocity = math.copysign(self.turn_speed, angle_rad)
        duration = abs(angle_rad) / max(self.turn_speed, 1e-6)
        hold_steps = self._duration_to_steps(duration)
        return ParsedPrimitive(np.array([0.0, 0.0, angular_velocity], dtype=np.float32), hold_steps, kind)

    def _move(self, distance_m: float, kind: str) -> ParsedPrimitive:
        linear_velocity = math.copysign(self.forward_speed, distance_m)
        duration = abs(distance_m) / max(self.forward_speed, 1e-6)
        hold_steps = self._duration_to_steps(duration)
        return ParsedPrimitive(np.array([linear_velocity, 0.0, 0.0], dtype=np.float32), hold_steps, kind)

    def _duration_to_steps(self, duration_s: float) -> int:
        return max(1, min(self.max_hold_steps, int(math.ceil(duration_s / self.policy_dt))))


class LocalNaVILAInference:
    """Small wrapper around NaVILA's run_navigation inference path."""

    def __init__(
        self,
        navila_repo: str,
        model_path: str = DEFAULT_NAVILA_MODEL,
        model_base: str | None = None,
        device: str = "cuda",
        device_map: str = "auto",
        load_4bit: bool = False,
        load_8bit: bool = False,
        conv_mode: str = "llama_3",
        max_new_tokens: int = 96,
    ) -> None:
        self.navila_repo = os.path.abspath(os.path.expanduser(navila_repo))
        if not os.path.isdir(self.navila_repo):
            raise FileNotFoundError(f"NaVILA repository was not found: {self.navila_repo}")
        if self.navila_repo not in sys.path:
            sys.path.insert(0, self.navila_repo)

        try:
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images
            from llava.mm_utils import tokenizer_image_token
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
        except Exception as exc:
            raise RuntimeError(
                "Failed to import NaVILA. Install NaVILA/VILA dependencies in the Python environment that runs "
                "Isaac Lab, or set --vla_repo to the local NaVILA-main path."
            ) from exc

        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SeparatorStyle = SeparatorStyle
        self.conv_templates = conv_templates
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.conv_mode = conv_mode
        self.max_new_tokens = int(max_new_tokens)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        load_kwargs = {}
        if load_4bit or load_8bit:
            load_kwargs["torch_dtype"] = torch.float16
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_name,
            model_base,
            device=device,
            device_map=device_map,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            **load_kwargs,
        )
        self.model_device = next(self.model.parameters()).device
        self.image_dtype = torch.float16 if self.model_device.type == "cuda" else torch.float32

    def generate(self, images: list[Image.Image], instruction: str) -> str:
        if not images:
            raise ValueError("At least one image is required for NaVILA inference.")

        conv = self.conv_templates[self.conv_mode].copy()
        image_token = "<image>\n"
        question = (
            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations {image_token * (len(images) - 1)}, and current observation <image>\n. "
            f'Your assigned task is: "{instruction}" '
            "Analyze this series of images to decide your next action, which could be turning left or right by a "
            "specific degree, moving forward a certain distance, or stop if the task is completed. "
            "Return only one concise command: turn left N degrees, turn right N degrees, move forward N cm, "
            "or stop."
        )
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = self.process_images(images, self.image_processor, self.model.config)
        images_tensor = images_tensor.to(self.model_device, dtype=self.image_dtype)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.model_device)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        stopping_criteria = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if output.endswith(stop_str):
            output = output[: -len(stop_str)]
        return output.strip()


class NaVILAVelocityPolicy:
    """Callable policy that plugs NaVILA into VR-Robo's 3D high-level action interface."""

    def __init__(
        self,
        env: Any,
        navila_repo: str,
        model_path: str = DEFAULT_NAVILA_MODEL,
        model_base: str | None = None,
        instruction: str | None = None,
        instruction_template: str = "Navigate to the {target}.",
        target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
        camera_name: str = "front_rgb_camera",
        camera_data_type: str = "rgb",
        num_video_frames: int = 8,
        env_id: int = 0,
        update_every: int = 1,
        forward_speed: float = 0.5,
        turn_speed: float = 0.8,
        default_forward_distance: float = 0.25,
        default_turn_degrees: float = 30.0,
        max_hold_steps: int = 60,
        device: str = "cuda",
        device_map: str = "auto",
        load_4bit: bool = False,
        load_8bit: bool = False,
        conv_mode: str = "llama_3",
        max_new_tokens: int = 96,
        log_path: str | None = None,
    ) -> None:
        self.env = env
        self.env_id = int(env_id)
        self.camera_name = camera_name
        self.camera_data_type = camera_data_type
        self.num_video_frames = max(1, int(num_video_frames))
        self.frames: deque[Image.Image] = deque(maxlen=self.num_video_frames)
        self.instruction = instruction
        self.instruction_template = instruction_template
        self.target_names = target_names
        self.update_every = max(1, int(update_every))
        self.update_counter = 0
        self.hold_steps_remaining = 0
        self.last_raw_action: torch.Tensor | None = None
        self.last_velocity_command = torch.zeros(env.num_envs, 3, device=env.device)
        self.last_response = ""
        self.last_primitive = "init"
        self.step_index = 0
        self.log_path = log_path
        self.log_file = None
        self.log_writer = None

        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        self.velocity_range = action_term.velocity_range.detach().clone().to(env.device)
        policy_dt = float(getattr(env.unwrapped, "step_dt", 0.0))
        if policy_dt <= 0.0:
            sim_dt = float(getattr(env.unwrapped.cfg.sim, "dt", 0.0))
            decimation = int(getattr(env.unwrapped.cfg, "decimation", 1))
            policy_dt = sim_dt * decimation
        self.parser = NaVILAActionParser(
            policy_dt=policy_dt,
            forward_speed=forward_speed,
            turn_speed=turn_speed,
            default_forward_distance=default_forward_distance,
            default_turn_degrees=default_turn_degrees,
            max_hold_steps=max_hold_steps,
        )
        self.model = LocalNaVILAInference(
            navila_repo=navila_repo,
            model_path=model_path,
            model_base=model_base,
            device=device,
            device_map=device_map,
            load_4bit=load_4bit,
            load_8bit=load_8bit,
            conv_mode=conv_mode,
            max_new_tokens=max_new_tokens,
        )
        if self.log_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            self.log_file = open(self.log_path, "w", newline="")
            self.log_writer = csv.writer(self.log_file)
            self.log_writer.writerow(
                [
                    "step",
                    "instruction",
                    "response",
                    "primitive",
                    "hold_steps",
                    "cmd_vx",
                    "cmd_vy",
                    "cmd_wz",
                ]
            )

    def __call__(self, _obs: torch.Tensor) -> torch.Tensor:
        if self.last_raw_action is not None:
            if self.hold_steps_remaining > 0:
                self.hold_steps_remaining -= 1
                self.step_index += 1
                return self.last_raw_action.clone()
            self.update_counter += 1
            if self.update_counter < self.update_every:
                self.step_index += 1
                return self.last_raw_action.clone()

        frame = self._capture_frame()
        self.frames.append(frame)
        while len(self.frames) < self.num_video_frames:
            self.frames.append(frame.copy())

        instruction = self._resolve_instruction()
        response = self.model.generate(list(self.frames), instruction)
        primitive = self.parser.parse(response)
        self.last_response = response
        self.last_primitive = primitive.kind
        self.hold_steps_remaining = max(0, primitive.hold_steps - 1)
        self.update_counter = 0
        self.last_velocity_command = torch.tensor(primitive.velocity_command, device=self.env.device).repeat(
            self.env.num_envs, 1
        )
        self.last_raw_action = command_to_raw_action(self.last_velocity_command, self.velocity_range)
        self._log(instruction, response, primitive)
        print(
            "[NaVILA] "
            f"response='{response}' primitive={primitive.kind} "
            f"cmd={primitive.velocity_command.tolist()} hold_steps={primitive.hold_steps}"
        )
        self.step_index += 1
        return self.last_raw_action.clone()

    def close(self) -> None:
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def _resolve_instruction(self) -> str:
        if self.instruction:
            return self.instruction

        try:
            command = self.env.unwrapped.command_manager.get_command("rgb_command")
            command_env = command[self.env_id].detach().cpu().numpy()
            target_idx = int(np.argmax(command_env))
            target = self.target_names[target_idx] if target_idx < len(self.target_names) else "target object"
            return self.instruction_template.format(target=target)
        except Exception:
            return self.instruction_template.format(target="target object")

    def _capture_frame(self) -> Image.Image:
        try:
            sensor = self.env.unwrapped.scene.sensors[self.camera_name]
            image = sensor.data.output[self.camera_data_type][self.env_id]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read {self.camera_data_type!r} from camera sensor {self.camera_name!r}. "
                "Run a camera task such as mclquad_seminar_camera_play and keep cameras enabled."
            ) from exc
        return self._to_pil_image(image)

    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        if torch.is_tensor(image):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
                image = image.permute(1, 2, 0)
            array = image.numpy()
        else:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                array = np.moveaxis(array, 0, -1)

        if array.ndim != 3:
            raise ValueError(f"Expected an RGB image with 3 dimensions, got shape {array.shape}")
        if array.shape[-1] > 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            if float(np.max(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def _log(self, instruction: str, response: str, primitive: ParsedPrimitive) -> None:
        if self.log_writer is None:
            return
        self.log_writer.writerow(
            [
                self.step_index,
                instruction,
                response,
                primitive.kind,
                primitive.hold_steps,
                *primitive.velocity_command.tolist(),
            ]
        )
        self.log_file.flush()
