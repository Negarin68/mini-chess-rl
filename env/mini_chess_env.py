# env/mini_chess_env.py

import numpy as np

class MiniChessEnv:
    """
    Simple 4x4 mini-chess-like environment.

    Pieces:
        White: King (WK), Rook (WR)
        Black: King (BK)

    Board encoding:
        0: empty
        1: white king
        2: white rook
        3: black king

    State shape: (4, 4, 4)  -> channels: WK, WR, BK, turn
    Action space: 256 (from_square * 16 + to_square)
    """

    BOARD_SIZE = 4
    MAX_STEPS = 40

    EMPTY = 0
    WK = 1
    WR = 2
    BK = 3

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int32)
        self.turn_white = True
        self.step_count = 0
        self.done = False

    def reset(self):
        """
        Reset the board to a simple fixed starting position.
        You can randomize later if خواستی.
        """
        self.board[:] = self.EMPTY
        self.turn_white = True
        self.step_count = 0
        self.done = False

        # Fixed simple layout:
        # White king at (3, 0), white rook at (3, 1), black king at (0, 3)
        self.board[3, 0] = self.WK
        self.board[3, 1] = self.WR
        self.board[0, 3] = self.BK

        return self._get_state()

    def _get_state(self):
        """
        Convert board to (4, 4, 4) tensor:
        channel 0: WK, 1: WR, 2: BK, 3: turn indicator (1 for white, 0 for black)
        """
        h, w = self.board.shape
        state = np.zeros((h, w, 4), dtype=np.float32)

        state[:, :, 0] = (self.board == self.WK).astype(np.float32)
        state[:, :, 1] = (self.board == self.WR).astype(np.float32)
        state[:, :, 2] = (self.board == self.BK).astype(np.float32)
        state[:, :, 3] = 1.0 if self.turn_white else 0.0

        return state

    def _index_to_pos(self, idx: int):
        """Convert 0..15 to (row, col)."""
        row = idx // self.BOARD_SIZE
        col = idx % self.BOARD_SIZE
        return row, col

    def _is_on_board(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def legal_actions(self):
        """
        Return list of legal action indices for the current player.
        Very simplified movement rules:
            - WK moves 1 step in 8 directions
            - WR moves horizontally/vertically any distance
        No check/checkmate rules yet – فقط جلوگیری از حرکت غیرمنطقی.
        """
        actions = []
        # find pieces
        wk_pos = None
        wr_pos = None
        bk_pos = None

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                p = self.board[r, c]
                if p == self.WK:
                    wk_pos = (r, c)
                elif p == self.WR:
                    wr_pos = (r, c)
                elif p == self.BK:
                    bk_pos = (r, c)

        if self.turn_white:
            # White moves WK or WR
            if wk_pos is not None:
                r, c = wk_pos
                directions = [(-1, -1), (-1, 0), (-1, 1),
                              (0, -1),          (0, 1),
                              (1, -1),  (1, 0), (1, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if self._is_on_board(nr, nc):
                        if self.board[nr, nc] != self.WK and self.board[nr, nc] != self.WR:
                            from_idx = r * self.BOARD_SIZE + c
                            to_idx = nr * self.BOARD_SIZE + nc
                            actions.append(from_idx * 16 + to_idx)

            if wr_pos is not None:
                r, c = wr_pos
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    while self._is_on_board(nr, nc):
                        if self.board[nr, nc] == self.WK or self.board[nr, nc] == self.WR:
                            break  # own piece
                        from_idx = r * self.BOARD_SIZE + c
                        to_idx = nr * self.BOARD_SIZE + nc
                        actions.append(from_idx * 16 + to_idx)
                        if self.board[nr, nc] != self.EMPTY:
                            break  # stop at capture
                        nr += dr
                        nc += dc
        else:
            # Black simple random king move (environment-controlled, not RL agent)
            if bk_pos is not None:
                r, c = bk_pos
                directions = [(-1, -1), (-1, 0), (-1, 1),
                              (0, -1),          (0, 1),
                              (1, -1),  (1, 0), (1, 1)]
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if self._is_on_board(nr, nc):
                        if self.board[nr, nc] != self.BK:
                            from_idx = r * self.BOARD_SIZE + c
                            to_idx = nr * self.BOARD_SIZE + nc
                            actions.append(from_idx * 16 + to_idx)

        return actions

    def step(self, action: int):
        """
        Take one step in the environment using action index.
        Returns: next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Call reset() before calling step() again.")

        self.step_count += 1
        reward = -0.01  # small step penalty
        info = {}

        legal = self.legal_actions()
        if action not in legal:
            # illegal move: punish and end
            reward = -1.0
            self.done = True
            return self._get_state(), reward, self.done, info

        from_idx = action // 16
        to_idx = action % 16
        fr, fc = self._index_to_pos(from_idx)
        tr, tc = self._index_to_pos(to_idx)

        moving_piece = self.board[fr, fc]
        target_piece = self.board[tr, tc]

        # move the piece
        self.board[fr, fc] = self.EMPTY
        self.board[tr, tc] = moving_piece

        # check capture
        if target_piece == self.BK:
            # captured black king → white wins
            reward = 1.0
            self.done = True

        if self.step_count >= self.MAX_STEPS and not self.done:
            # draw
            reward = 0.0
            self.done = True

        # if game not over, let black move randomly
        if not self.done:
            self.turn_white = False
            black_legal = self.legal_actions()
            if black_legal:
                b_action = self.rng.choice(black_legal)
                b_from = b_action // 16
                b_to = b_action % 16
                b_fr, b_fc = self._index_to_pos(b_from)
                b_tr, b_tc = self._index_to_pos(b_to)

                moving_piece = self.board[b_fr, b_fc]
                target_piece = self.board[b_tr, b_tc]
                self.board[b_fr, b_fc] = self.EMPTY
                self.board[b_tr, b_tc] = moving_piece

                if target_piece == self.WK:
                    # black captures white king → loss
                    reward = -1.0
                    self.done = True

            self.turn_white = True

        next_state = self._get_state()
        return next_state, reward, self.done, info
