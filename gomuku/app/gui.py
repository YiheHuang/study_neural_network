from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import torch

from env.board import Board, GameResult, Player
from mcts import MCTS
from model.network import GomokuNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gomoku GUI (Tkinter)")
    parser.add_argument("--board-size", type=int, default=15)
    parser.add_argument("--cell-size", type=int, default=36)
    parser.add_argument("--human", choices=["black", "white"], default="black")
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=6)
    return parser.parse_args()


class GomokuGUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.board = Board(size=args.board_size)
        self.human_player = Player.BLACK if args.human == "black" else Player.WHITE
        self.ai_player = self.human_player.opponent()
        self.ai_thinking = False

        self.model = GomokuNet(
            board_size=args.board_size,
            channels=args.channels,
            num_res_blocks=args.res_blocks,
        )
        self._try_load_model(args.model_path)
        self.mcts = MCTS(self.model, c_puct=1.5)

        self.margin = 24
        self.cell = args.cell_size
        board_px = self.margin * 2 + self.cell * (args.board_size - 1)

        self.root = tk.Tk()
        self.root.title("Gomoku AI")
        self.status_var = tk.StringVar()
        self.status_var.set("准备开始")

        top = tk.Frame(self.root)
        top.pack(fill="x")
        tk.Label(top, textvariable=self.status_var, anchor="w").pack(side="left", padx=8, pady=6)
        tk.Button(top, text="重新开始", command=self.reset).pack(side="right", padx=8, pady=6)

        self.canvas = tk.Canvas(self.root, width=board_px, height=board_px, bg="#E8B66B")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_board()
        self.update_status()
        self.maybe_ai_turn()

    def _try_load_model(self, model_path: str) -> None:
        chosen = model_path
        if not chosen:
            for candidate in ("checkpoints/best_model.pt", "checkpoints/latest_model.pt"):
                if Path(candidate).exists():
                    chosen = candidate
                    break
        if chosen:
            state = torch.load(chosen, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.model.load_state_dict(state)

    def reset(self) -> None:
        if self.ai_thinking:
            return
        self.board.reset()
        self.draw_board()
        self.update_status()
        self.maybe_ai_turn()

    def board_to_canvas(self, idx: int) -> int:
        return self.margin + idx * self.cell

    def canvas_to_board(self, value: int) -> int:
        return round((value - self.margin) / self.cell)

    def draw_board(self) -> None:
        self.canvas.delete("all")
        n = self.board.size
        for i in range(n):
            x0 = self.board_to_canvas(0)
            x1 = self.board_to_canvas(n - 1)
            y = self.board_to_canvas(i)
            self.canvas.create_line(x0, y, x1, y, fill="black")
            x = self.board_to_canvas(i)
            y0 = self.board_to_canvas(0)
            y1 = self.board_to_canvas(n - 1)
            self.canvas.create_line(x, y0, x, y1, fill="black")
        for r in range(n):
            for c in range(n):
                stone = int(self.board.grid[r, c])
                if stone == int(Player.EMPTY):
                    continue
                self.draw_stone(r, c, Player(stone))

    def draw_stone(self, row: int, col: int, player: Player) -> None:
        x = self.board_to_canvas(col)
        y = self.board_to_canvas(row)
        radius = self.cell // 2 - 2
        color = "black" if player == Player.BLACK else "white"
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color)

    def on_click(self, event: tk.Event) -> None:
        if self.ai_thinking or self.board.game_result() != GameResult.ONGOING:
            return
        if self.board.current_player != self.human_player:
            return

        row = self.canvas_to_board(event.y)
        col = self.canvas_to_board(event.x)
        if not self.board.is_legal_move(row, col):
            return
        self.board.place_stone(row, col)
        self.draw_board()
        self.update_status()
        self.maybe_ai_turn()

    def maybe_ai_turn(self) -> None:
        if self.board.game_result() != GameResult.ONGOING:
            self.show_game_over()
            return
        if self.board.current_player != self.ai_player:
            return
        self.ai_thinking = True
        self.status_var.set("AI 思考中...")
        threading.Thread(target=self._ai_move_worker, daemon=True).start()

    def _ai_move_worker(self) -> None:
        try:
            move, _ = self.mcts.run(
                self.board,
                simulations=self.args.simulations,
                device=self.args.device,
            )
            self.root.after(0, lambda: self._finish_ai_move(move))
        except Exception as exc:
            self.root.after(0, lambda: self._handle_ai_error(str(exc)))

    def _finish_ai_move(self, move: tuple[int, int]) -> None:
        if self.board.game_result() == GameResult.ONGOING:
            self.board.place_stone(*move)
        self.ai_thinking = False
        self.draw_board()
        self.update_status()
        self.maybe_ai_turn()

    def _handle_ai_error(self, err: str) -> None:
        self.ai_thinking = False
        messagebox.showerror("AI Error", err)
        self.update_status()

    def update_status(self) -> None:
        result = self.board.game_result()
        if result == GameResult.ONGOING:
            side = "黑" if self.board.current_player == Player.BLACK else "白"
            self.status_var.set(f"当前回合: {side}")
            return
        if result == GameResult.DRAW:
            self.status_var.set("对局结束: 平局")
        elif result == GameResult.BLACK_WIN:
            self.status_var.set("对局结束: 黑胜")
        else:
            self.status_var.set("对局结束: 白胜")

    def show_game_over(self) -> None:
        result = self.board.game_result()
        if result == GameResult.DRAW:
            message = "平局"
        elif result == GameResult.BLACK_WIN:
            message = "黑方获胜"
        else:
            message = "白方获胜"
        messagebox.showinfo("对局结束", message)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    app = GomokuGUI(args)
    app.run()


if __name__ == "__main__":
    main()
