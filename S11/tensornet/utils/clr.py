import matplotlib.pyplot as plt


class CyclicLR:

    def __init__(self, lr_max, lr_min, step_size, num_iterations):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_size = step_size
        self.iterations = num_iterations
        self.lr = []
        self.pad_factor = lr_max / 10

    def cycle(self, iteration):
        return int(1 + (iteration / (2 * self.step_size)))

    def lr_position(self, iteration, cycle):
        return abs(iteration / self.step_size - 2 * cycle + 1)

    def current_lr(self, lr_position):
        return self.lr_min + (self.lr_max - self.lr_min) * (1 - lr_position)

    def cyclic_lr(self, plot=True):
        for iteration in range(self.iterations):
            cycle = self.cycle(iteration)
            lr_position = self.lr_position(iteration, cycle)
            self.lr.append(self.current_lr(lr_position))
        if plot:
            self.plot()
    
    def plot(self):
        # Initialize a figure
        fig = plt.figure(figsize=(10, 3))

        # Set plot title
        plt.title('Cyclic LR')

        # Label axes
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')

        # Plot max lr line
        plt.axhline(self.lr_max, 0.03, 0.97, label='max_lr', color='y')
        plt.text(0, self.lr_max + self.pad_factor, 'max_lr')

        # Plot min lr line
        plt.axhline(self.lr_min, 0.03, 0.97, label='min_lr', color='y')
        plt.text(0, self.lr_min - self.pad_factor, 'min_lr')

        # Plot lr change
        plt.plot(self.lr)

        # Plot margins and save plot
        plt.margins(y=0.2)
        plt.tight_layout()
        plt.savefig('clr_plot.png')
