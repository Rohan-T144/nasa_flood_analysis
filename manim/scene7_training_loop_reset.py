# scene7_training_loop_reset_v0_19_fixed.py
from manim import *

class TrainingLoopResetScene(Scene):
    def construct(self):
        # Narration: "A key difference in training SNNs is managing their stateful nature."
        title = Text("SNN Training and State Reset").to_edge(UP)
        self.play(Write(title))

        # --- Setup ---
        # A VGroup to hold the model and its labels, for easy positioning.
        model_group = VGroup()
        
        model_rect = Rectangle(width=3, height=3.5, color=BLUE)
        neuron_states = VGroup(
            Circle(radius=0.2, color=YELLOW, fill_opacity=0.0).shift(UP*1.2 + LEFT*0.7),
            Circle(radius=0.2, color=YELLOW, fill_opacity=0.0).shift(DOWN*0.6 + RIGHT*0.3),
            Circle(radius=0.2, color=YELLOW, fill_opacity=0.0).shift(RIGHT*0.7),
        )
        model_vis = VGroup(model_rect, neuron_states)
        
        model_label = Text("SpikingUNet Model", font_size=24).next_to(model_vis, DOWN, buff=0.3)
        state_label = Text("Internal Membrane Potentials", font_size=20).next_to(model_vis, UP, buff=0.3)
        
        model_group.add(model_vis, model_label, state_label).center()
        
        # --- Create a dedicated status text area ---
        # This prevents text from overlapping below the model.
        status_text = Text(".", font_size=24).next_to(model_group, UP, buff=0.2)

        self.play(Create(model_vis), Write(model_label), Write(state_label))
        self.wait(1)

        # --- Animation ---
        
        # Process Batch 1
        batch1 = Text("Batch 1", font_size=30).to_edge(LEFT)
        self.play(Write(batch1))
        
        # Animate batch moving towards the model
        self.play(batch1.animate.move_to(model_vis.get_left() + LEFT))

        # Narration: "When Batch 1 of data enters... the internal membrane potentials of the neurons change..."
        self.play(
            status_text.animate.become(Text("Processing Batch 1...", font_size=24).move_to(status_text)),
            batch1.animate.move_to(model_vis.get_center()),
            run_time=1
        )
        self.play(
            FadeOut(batch1),
            neuron_states[0].animate.set_fill(YELLOW, opacity=0.7),
            neuron_states[1].animate.set_fill(YELLOW, opacity=0.2),
            neuron_states[2].animate.set_fill(YELLOW, opacity=0.5),
            status_text.animate.become(Text("Forward / Backward Pass", font_size=24).move_to(status_text)),
            run_time=2
        )
        self.wait(1)

        # Narration: "Because SNNs have memory, we must reset the state of all neurons after processing each batch."
        reset_command_code = Text("functional.reset_net(model)", font="Monospace", font_size=24).next_to(model_vis, DOWN, buff=0.7)#.next_to(title, DOWN*0.3, buff=0.5)
        reset_flash = Text("RESET", font_size=96, color=RED, weight=BOLD)
        
        self.play(
            FadeOut(status_text),
            FadeIn(reset_command_code, shift=DOWN)
        )
        self.play(Flash(reset_flash, flash_radius=2, time_width=1))
        
        # Narration: "This instantly resets all membrane potentials to zero."
        self.play(
            *[neuron.animate.set_fill(YELLOW, opacity=0) for neuron in neuron_states],
            status_text.animate.become(Text("Resetting state...", font_size=24).move_to(status_text)),
            run_time=1
        )
        self.wait(1)

        # Process Batch 2
        # Narration: "Now the network is clean, ready to process Batch 2..."
        self.play(FadeOut(reset_command_code, status_text))
        
        batch2 = Text("Batch 2", font_size=30).to_edge(LEFT)
        self.play(Write(batch2))
        self.play(batch2.animate.move_to(model_vis.get_left() + LEFT))

        self.play(
            status_text.animate.become(Text("Processing Batch 2...", font_size=24).move_to(status_text)),
            batch2.animate.move_to(model_vis.get_center()),
            run_time=1
        )
        self.play(
            FadeOut(batch2),
            neuron_states[0].animate.set_fill(YELLOW, opacity=0.6),
            neuron_states[1].animate.set_fill(YELLOW, opacity=0.3),
            neuron_states[2].animate.set_fill(YELLOW, opacity=0.8),
            run_time=2
        )
        self.play(FadeOut(status_text))
        
        self.wait(2)
