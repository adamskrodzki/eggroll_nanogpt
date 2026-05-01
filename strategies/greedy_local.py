from .greedy import GreedyEGGROLL


class GreedyLocalEGGROLL(GreedyEGGROLL):
    def compute_update(self, linear_layers, fitness, avg_loss):
        best_loss = avg_loss.min().item()
        if best_loss < self.best_loss_so_far:
            super().compute_update(linear_layers, fitness, avg_loss)
        else:
            print(f"Best loss {best_loss} is not better than best loss so far {self.best_loss_so_far}, skipping update")
