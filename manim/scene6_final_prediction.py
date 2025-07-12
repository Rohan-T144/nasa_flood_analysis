# scene6_final_prediction_v0_19_resized.py
from manim import *

class FinalPredictionScene(Scene):
    def construct(self):
        # --- Overall Scaling and Layout ---
        scale = 0.8  # Master scale factor for all elements
        h_buff = 0.8 # Horizontal buffer between elements
        
        # Narration: "After passing through the full U-Net, the final decoder block produces a spike train."
        title = Text("Generating the Final Prediction").to_edge(UP)
        self.play(Write(title))

        # --- Define All Components First ---

        # 1. Spike Train Input
        decoder_output = VGroup(*[
            Square(side_length=1.2*scale, color=BLUE, fill_opacity=0.3) for _ in range(4)
        ]).arrange(RIGHT, buff=0.2*scale)
        decoder_label = Text("Spike Train", font_size=20*scale).next_to(decoder_output, UP, buff=0.2*scale)
        decoder_group = VGroup(decoder_output, decoder_label)

        # 2. Final LIF Layer
        final_lif = Rectangle(width=1.8*scale, height=1.8*scale, color=GREEN, fill_opacity=0.2)
        final_lif_label = Text("final_lif", font_size=20*scale).next_to(final_lif, DOWN, buff=0.2*scale)
        lif_group = VGroup(final_lif, final_lif_label)
        
        # 3. Summed Potential
        summed_potential = Rectangle(width=1.8*scale, height=1.8*scale, color=PURPLE, fill_opacity=0.5)
        summed_label = Text("Summed Potential", font_size=20*scale).next_to(summed_potential, DOWN, buff=0.2*scale)
        summed_group = VGroup(summed_potential, summed_label)

        # 4. Final Convolution
        final_conv = Rectangle(width=1.8*scale, height=1.8*scale, color=ORANGE)
        conv_label = Text("OutConv\n+ Sigmoid", font_size=18*scale, line_spacing=1).move_to(final_conv)
        conv_group = VGroup(final_conv, conv_label)

        # 5. Final Mask Output
        final_mask = Square(side_length=1.8*scale, color=WHITE, fill_opacity=1.0)
        inner_shape = Circle(radius=0.4*scale, color=BLACK, fill_opacity=1).move_to(final_mask)
        mask_content = VGroup(final_mask, inner_shape)
        mask_label = Text("Final Mask", font_size=20*scale).next_to(mask_content, DOWN, buff=0.2*scale)
        mask_group = VGroup(mask_content, mask_label)

        # --- Arrange Layout in Master VGroups ---
        # Top row for the input
        decoder_group.center().to_edge(UP, buff=1.5)
        
        # Bottom row for the processing pipeline
        pipeline = VGroup(
            lif_group, summed_group, conv_group, mask_group
        ).arrange(RIGHT, buff=h_buff).next_to(decoder_group, DOWN, buff=1.5)

        # --- ANIMATION SEQUENCE ---

        # Animate the spike train input
        self.play(Write(decoder_group))
        for block in decoder_output:
            self.play(Indicate(block, color=YELLOW), run_time=0.4)
        
        # Narration: "This train is fed into one last layer of LIF neurons."
        arrow_to_lif = Arrow(decoder_group.get_bottom(), lif_group.get_top(), buff=0.2, stroke_width=3*scale)
        self.play(GrowArrow(arrow_to_lif), FadeIn(lif_group))
        self.wait(1)

        # Narration: "Instead of their spikes, this time we care about their final membrane potential..."
        potential_text = Text("Membrane Potential", font_size=24*scale).move_to(final_lif)
        self.play(Write(potential_text))
        self.play(final_lif.animate.set_fill(color=YELLOW, opacity=0.8), run_time=2, rate_func=there_and_back)
        self.play(FadeOut(potential_text))
        
        # Narration: "The model then sums the potential across the time dimension."
        sum_op_code = Text("potential.sum(dim=0)", font="Monospace", font_size=18*scale).next_to(lif_group, UP, buff=0.2)
        arrow_to_sum = Arrow(lif_group.get_right(), summed_group.get_left(), buff=0.2, stroke_width=3*scale)
        self.play(Write(sum_op_code))
        self.play(GrowArrow(arrow_to_sum), FadeIn(summed_group, shift=RIGHT))
        self.wait(1)

        # Narration: "This single tensor... is passed through a final convolution and a sigmoid function..."
        arrow_to_conv = Arrow(summed_group.get_right(), conv_group.get_left(), buff=0.2, stroke_width=3*scale)
        self.play(GrowArrow(arrow_to_conv), FadeIn(conv_group, shift=RIGHT))
        self.wait(1)
        
        # Narration: "...to produce the final black and white segmentation mask."
        arrow_to_mask = Arrow(conv_group.get_right(), mask_group.get_left(), buff=0.2, stroke_width=3*scale)
        self.play(GrowArrow(arrow_to_mask), FadeIn(mask_group, shift=RIGHT))
        self.wait(2)
