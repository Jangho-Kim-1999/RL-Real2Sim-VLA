from __future__ import annotations

import argparse
import pickle
import socket
import threading
import time
from pathlib import Path

import numpy as np
import rpyc
import torch
from PIL import Image


rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True


def receive_tensor(host: str, port: int, timeout_s: float) -> np.ndarray:
    chunks: list[bytes] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout_s)
        sock.bind((host, port))
        sock.listen(1)
        conn, _ = sock.accept()
        with conn:
            conn.settimeout(timeout_s)
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                chunks.append(packet)
    return pickle.loads(b"".join(chunks))


def tensor_to_images(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 2 or tensor.shape[1] != 3 * 180 * 320:
        raise ValueError(f"Unexpected render tensor shape: {tensor.shape}")
    return tensor.reshape(tensor.shape[0], 3, 180, 320).transpose(0, 2, 3, 1)


def default_render_request(num_views: int):
    base_pos = torch.tensor(
        [
            [-2.0, -0.5, 0.5],
            [-1.0, 0.5, 0.8],
            [0.0, 1.5, 1.0],
            [1.0, 2.5, 1.0],
        ],
        dtype=torch.float32,
    )
    base_ori = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float32).repeat(base_pos.shape[0], 1)
    red = torch.tensor([[-0.2, -0.2, 0.36]], dtype=torch.float32).repeat(base_pos.shape[0], 1)
    green = torch.tensor([[-0.2, -0.6, 0.36]], dtype=torch.float32).repeat(base_pos.shape[0], 1)
    blue = torch.tensor([[-0.2, -1.0, 0.36]], dtype=torch.float32).repeat(base_pos.shape[0], 1)

    n = max(1, min(num_views, base_pos.shape[0]))
    return base_pos[:n], base_ori[:n], red[:n], green[:n], blue[:n]


def save_images(images: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        Image.fromarray(image.astype(np.uint8)).save(output_dir / f"render_{idx:02d}.png")


def print_stats(images: np.ndarray):
    for idx, image in enumerate(images):
        black_ratio = float(np.all(image < 5, axis=-1).mean())
        bright_ratio = float(np.any(image > 20, axis=-1).mean())
        print(
            f"image[{idx}] mean={image.mean():.2f} std={image.std():.2f} "
            f"min={image.min()} max={image.max()} black_ratio={black_ratio:.3f} "
            f"visible_ratio={bright_ratio:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Request one Gaussian-splat render and save returned RGB images.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--rpc_port", type=int, default=18861)
    parser.add_argument("--image_port", type=int, default=12345)
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--output_dir", default="render_debug")
    args = parser.parse_args()

    received: dict[str, np.ndarray | BaseException] = {}

    def listen():
        try:
            received["tensor"] = receive_tensor(args.host, args.image_port, args.timeout_s)
        except BaseException as exc:
            received["error"] = exc

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    time.sleep(0.2)

    conn = rpyc.connect(args.host, args.rpc_port, config={"allow_pickle": True, "allow_public_attrs": True})
    pos, ori, red, green, blue = default_render_request(args.num_views)
    conn.root.render(pos, ori, red, green, blue)

    listener.join(args.timeout_s + 2.0)
    if listener.is_alive():
        raise TimeoutError(f"Timed out waiting for tensor on {args.host}:{args.image_port}")
    if "error" in received:
        raise RuntimeError("Failed while receiving render tensor") from received["error"]

    images = tensor_to_images(received["tensor"])
    output_dir = Path(args.output_dir)
    save_images(images, output_dir)
    print(f"received tensor shape={received['tensor'].shape}, saved {images.shape[0]} image(s) to {output_dir}")
    print_stats(images)


if __name__ == "__main__":
    main()
