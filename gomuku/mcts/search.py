from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from env.board import Board, GameResult
from model.predict import predict_policy_value


def _move_to_index(board_size: int, move: Tuple[int, int]) -> int:
    return move[0] * board_size + move[1]


@dataclass
class MCTSNode:
    board: Board
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[Tuple[int, int], "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, model: torch.nn.Module, c_puct: float = 1.5) -> None:
        self.model = model
        self.c_puct = c_puct

    def run(
        self,
        root_board: Board,
        simulations: int = 200,
        device: str | torch.device = "cpu",
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        root = MCTSNode(board=root_board.copy(), prior=1.0)
        self._expand(root, device=device)
        if add_dirichlet_noise:
            self._apply_dirichlet_noise(
                root, alpha=dirichlet_alpha, epsilon=dirichlet_epsilon
            )

        for _ in range(simulations):
            node = root
            search_path: List[MCTSNode] = [node]

            while node.is_expanded and node.children:
                _, node = self._select_child(node)
                search_path.append(node)

            result = node.board.game_result()
            if result != GameResult.ONGOING:
                leaf_value = self._terminal_value(node.board, result)
            else:
                leaf_value = self._expand(node, device=device)

            self._backpropagate(search_path, leaf_value)

        move = self._select_action_by_temperature(root, temperature=temperature)
        policy = np.zeros(root.board.size * root.board.size, dtype=np.float32)
        visit_total = sum(child.visit_count for child in root.children.values())
        if visit_total > 0:
            for child_move, child in root.children.items():
                idx = _move_to_index(root.board.size, child_move)
                policy[idx] = child.visit_count / visit_total
        return move, policy

    def _apply_dirichlet_noise(
        self, node: MCTSNode, alpha: float, epsilon: float
    ) -> None:
        if not node.children:
            return
        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves)).astype(np.float32)
        for i, move in enumerate(moves):
            child = node.children[move]
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(noise[i])

    def _select_action_by_temperature(
        self, root: MCTSNode, temperature: float
    ) -> Tuple[int, int]:
        if not root.children:
            raise RuntimeError("Root node has no children.")
        if temperature <= 1e-6:
            move, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
            return move

        moves = list(root.children.keys())
        visits = np.array(
            [max(1e-6, float(root.children[m].visit_count)) for m in moves],
            dtype=np.float64,
        )
        scaled = np.power(visits, 1.0 / max(1e-6, temperature))
        probs = scaled / scaled.sum()
        idx = int(np.random.choice(len(moves), p=probs))
        return moves[idx]

    def _expand(self, node: MCTSNode, device: str | torch.device) -> float:
        priors, value = predict_policy_value(self.model, node.board, device=device)
        legal_moves = node.board.legal_moves()
        if not legal_moves:
            node.is_expanded = True
            return value

        for move in legal_moves:
            idx = _move_to_index(node.board.size, move)
            child_board = node.board.copy()
            child_board.place_stone(move[0], move[1])
            node.children[move] = MCTSNode(board=child_board, prior=float(priors[idx]))

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[Tuple[int, int], MCTSNode]:
        best_score = -float("inf")
        best_move: Tuple[int, int] | None = None
        best_child: MCTSNode | None = None
        parent_visits_sqrt = math.sqrt(max(1, node.visit_count))

        for move, child in node.children.items():
            q = -child.value()
            u = self.c_puct * child.prior * parent_visits_sqrt / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_move is None or best_child is None:
            raise RuntimeError("Failed to select child in MCTS.")
        return best_move, best_child

    def _backpropagate(self, search_path: List[MCTSNode], value: float) -> None:
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _terminal_value(self, board: Board, result: GameResult) -> float:
        if result == GameResult.DRAW:
            return 0.0
        if result == GameResult.BLACK_WIN:
            winner = 1
        elif result == GameResult.WHITE_WIN:
            winner = -1
        else:
            return 0.0
        # place_stone 在获胜时不切换 current_player，因此终局节点与父节点
        # 属于同一玩家。反向传播的符号交替假设每层玩家交替，所以需要返回
        # "假设下一个行动者"的视角值（即输家视角 = -1.0）。
        return -1.0 if int(board.current_player) == winner else 1.0
