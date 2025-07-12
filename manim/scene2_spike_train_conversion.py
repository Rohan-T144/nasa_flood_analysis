# scene2_spike_train_conversion_v0_19.py
from manim import *

class SpikeTrainConversionScene(Scene):
    def construct(self):
        # Narration: "Spiking Neural Networks process data over time. Our model begins by converting a single, static image into a sequence."
        
        # --- Setup ---
        input_grid = Square(side_length=2, color=BLUE).set_fill(BLUE, opacity=0.3)
        input_label = Text("Static Input Image [N, C, H, W]").scale(0.5).next_to(input_grid, DOWN)
        input_group = VGroup(input_grid, input_label).to_edge(LEFT, buff=1.5)

        self.play(Create(input_grid), Write(input_label))
        self.wait(1)

        code_text = Text("x.unsqueeze(0).repeat(T, 1, 1, 1, 1)", font="Monospace", font_size=28).to_edge(UP)
        # Narration: "The input tensor is unsqueezed and repeated along a new temporal dimension, T."
        self.play(Write(code_text))
        self.wait(1)

        # --- Animation ---
        T = 4
        temporal_grids = VGroup()
        target_pos_x = 0
        
        # Narration: "This creates a sequence of identical frames, which is then treated as a stream of input spikes for the network."
        for i in range(T):
            grid_copy = input_grid.copy()
            target_pos = RIGHT * (target_pos_x + i * 1.5) + DOWN * 1.5
            grid_copy.move_to(target_pos).scale(0.7)
            temporal_grids.add(grid_copy)
        
        self.play(
            FadeOut(input_group, shift=LEFT),
            LaggedStart(*[FadeIn(g, shift=UP) for g in temporal_grids], lag_ratio=0.5),
        )
        self.wait(0.5)

        seq_label = Text("Spike Train [T, N, C, H, W]", font_size=36).next_to(temporal_grids, UP, buff=0.5)
        self.play(Write(seq_label))

        # Animate the time counter
        time_counter_label = Text("Time:", font_size=30).next_to(temporal_grids, DOWN, buff=0.3)
        # UPDATED: Replaced deprecated Integer with DecimalNumber
        time_counter_val = DecimalNumber(1, num_decimal_places=0, font_size=30).next_to(time_counter_label, RIGHT)
        
        #pointer = Arrow(start=time_counter_val.get_bottom() + DOWN * 0.2, end=temporal_grids[0].get_top(), color=YELLOW)
        
        self.play(Write(time_counter_label), Write(time_counter_val))#, GrowArrow(pointer))
        self.wait(0.5)
        self.play(Indicate(temporal_grids[0], color=YELLOW, scale_factor=1.2), run_time=0.3)
        for i in range(1, T):
            self.play(
                time_counter_val.animate.set_value(i + 1),
                #pointer.animate.next_to(temporal_grids[i].get_top()),
                run_time=0.7
            )
            self.play(Indicate(temporal_grids[i], color=YELLOW, scale_factor=1.2), run_time=0.3)

        self.wait(2)
