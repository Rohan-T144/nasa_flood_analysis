# scene1_lif_neuron_v0_19.py
from manim import *

class LIFNeuronScene(Scene):
    def construct(self):
        # Narration: "This is a Leaky Integrate-and-Fire, or LIF, neuron, the fundamental building block of our Spiking Neural Network."
        
        # --- Setup ---
        neuron = Circle(radius=0.5, color=BLUE, fill_opacity=0.5).to_edge(LEFT, buff=1)
        neuron_label = Text("LIF Neuron").next_to(neuron, DOWN)

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.2, 0.5],
            x_length=7,
            y_length=4,
            axis_config={"color": WHITE},
        ).to_edge(RIGHT, buff=1)
        
        y_label = axes.get_y_axis_label("Membrane Potential (v)", edge=LEFT, direction=UP)
        graph_title = VGroup(axes.get_x_axis_label("Time"), y_label)

        threshold = 1.0
        threshold_line = DashedLine(
            axes.c2p(0, threshold), axes.c2p(10, threshold), color=RED
        )
        threshold_label = Text("Firing Threshold", font_size=24, color=RED).next_to(threshold_line, UP, buff=0.1)

        self.play(Create(neuron), Write(neuron_label))
        self.play(Create(axes), Create(graph_title), Create(threshold_line), Write(threshold_label))
        self.wait(1)

        # --- Animation Logic ---
        # Narration: "Like a biological neuron, it has a membrane potential. When it receives input spikes, its potential increases."
        
        time = ValueTracker(0)
        potential = 0.0
        graph_path = VGroup() # To store segments of the graph
        
        # Function to add a segment to the graph path
        def add_graph_segment(start_t, end_t, start_v, leak_rate=0.2):
            new_path = axes.plot(
                lambda t: start_v * np.exp(-leak_rate * (t - start_t)),
                x_range=[start_t, end_t],
                color=YELLOW
            )
            graph_path.add(new_path)
            return new_path

        # Initial state
        dot = Dot(point=axes.c2p(0, 0), color=YELLOW)
        self.add(dot)

        # Move time forward to 1.5s (no spikes)
        segment1 = add_graph_segment(0, 1.5, potential)
        self.play(MoveAlongPath(dot, segment1), time.animate.set_value(1.5), run_time=1.5, rate_func=linear)
        potential = axes.point_to_coords(dot.get_center())[1]
        
        
        # Spike 1 arrives at t=1.5s
        spike1 = Dot(color=GREEN, radius=0.1).move_to(neuron.get_center() + LEFT * 3)
        self.play(spike1.animate.move_to(neuron), run_time=0.5)
        potential += 0.6
        self.play(FadeOut(spike1), Flash(neuron, color=GREEN, flash_radius=0.7), dot.animate.move_to(axes.c2p(1.5, potential)))
        
        # Narration: "If no spikes arrive, the potential 'leaks', slowly decaying back to its resting state."
        # Move time forward to 3.5s (leaking)
        segment2 = add_graph_segment(1.5, 3.5, potential)
        self.play(MoveAlongPath(dot, segment2), time.animate.set_value(3.5), run_time=2, rate_func=linear)
        potential = axes.point_to_coords(dot.get_center())[1]

        # Spike 2 & 3 arrive
        spike2 = Dot(color=GREEN, radius=0.1).move_to(neuron.get_center() + LEFT * 3)
        self.play(spike2.animate.move_to(neuron), run_time=0.5)
        potential += 0.4
        self.play(FadeOut(spike2), Flash(neuron, color=GREEN, flash_radius=0.7), dot.animate.move_to(axes.c2p(3.5, potential)))
        
        spike3 = Dot(color=GREEN, radius=0.1).move_to(neuron.get_center() + LEFT * 3 + UP * 0.5)
        self.play(spike3.animate.move_to(neuron), run_time=0.5)
        potential += 0.4 # This should cross the threshold
        self.play(FadeOut(spike3), Flash(neuron, color=GREEN, flash_radius=0.7), dot.animate.move_to(axes.c2p(3.5, potential)))

        # Narration: "When the potential crosses the firing threshold..."
        self.wait(0.5)

        # Firing
        # Narration: "...the neuron fires an output spike of its own..."
        output_spike = Dot(color=ORANGE, radius=0.15).move_to(neuron)
        self.play(Flash(neuron, color=ORANGE, flash_radius=1.0, line_length=0.5, num_lines=20, time_width=0.5))
        self.play(output_spike.animate.move_to(neuron.get_center() + RIGHT * 3))
        self.play(FadeOut(output_spike))
        
        # Narration: "...and its potential immediately resets. This behavior allows the network to process information through time."
        potential = 0
        self.play(dot.animate.move_to(axes.c2p(3.5, potential)), run_time=0.2)

        # Final leak
        segment3 = add_graph_segment(3.5, 8, potential)
        self.play(MoveAlongPath(dot, segment3), time.animate.set_value(8), run_time=4.5, rate_func=linear)
        
        self.wait(2)
