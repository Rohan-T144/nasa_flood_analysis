# scene3_spiking_double_conv_v0_19_fixed.py
from manim import *
import random
class SpikingDoubleConvScene(ThreeDScene):
    def construct(self):
        # Narration: "Inside the network, data flows through blocks like SpikingDoubleConv."
        
        # --- Setup ---
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        def create_spike_cube(dims, color):
            c, h, w = dims
            # Scale down the representation for visual clarity
            return VGroup(*[
                Dot3D(radius=0.05, color=color, resolution=(4,4)).move_to([i*0.2-w*0.1, j*0.2-h*0.1, k*0.1-c*0.05])
                for i in range(w) for j in range(h) for k in range(c)
            ]).scale(1.5)

        input_cube = create_spike_cube((3, 10, 10), BLUE).shift(LEFT * 4)
        
        # CORRECTED: .fix_in_frame() is removed from here...
        input_label = Text("Input Spikes", font_size=24).to_corner(UL)
        # ...and the mobject is added using the scene's method instead.
        self.add_fixed_in_frame_mobjects(input_label)
        
        self.play(Write(input_label), FadeIn(input_cube))

        # --- Animation ---
        # Narration: "This block processes spike trains over several timesteps."
        time_counter_val = DecimalNumber(1, num_decimal_places=0, font_size=24)
        time_counter_label = Text("Time: ", font_size=24)
        
        # CORRECTED: .fix_in_frame() is removed from here...
        time_counter = VGroup(time_counter_label, time_counter_val).arrange(RIGHT).to_corner(UR)
        # ...and the mobject is added using the scene's method.
        self.add_fixed_in_frame_mobjects(time_counter)
        self.play(Write(time_counter))

        def flash_spikes(cube, num_flashes=3):
            animations = []
            for i in range(num_flashes):
                # Animate the counter value changing
                update_counter = time_counter_val.animate.increment_value()
                dots_to_flash = random.sample(cube.submobjects, k=20)
                flash_anim = LaggedStart(*[
                    Flash(dot, color=YELLOW, time_width=0.3) for dot in dots_to_flash
                ], lag_ratio=0.05, run_time=0.5)
                
                # Run the counter update and the flash simultaneously
                animations.append(AnimationGroup(update_counter, flash_anim))
            return Succession(*animations)

        self.play(flash_spikes(input_cube, num_flashes=2))
        self.wait(0.5)

        # Narration: "First, a 3x3 convolution processes the input spikes."
        kernel = Cube(side_length=0.4, fill_opacity=0.2, color=RED).move_to(input_cube.get_center() + LEFT*0.7 + UP*0.7)
        self.play(Create(kernel))
        self.play(kernel.animate.move_to(input_cube.get_center() + RIGHT*0.7 + DOWN*0.7), run_time=2)
        
        # Narration: "The result is fed into a layer of LIF neurons."
        lif1_cube = create_spike_cube((5, 10, 10), GREEN)
        arrow1 = Arrow3D(input_cube.get_right(), lif1_cube.get_left(), color=WHITE, resolution=12)
        self.play(Create(arrow1), FadeIn(lif1_cube))

        # Narration: "These neurons integrate the inputs, and only fire if their potential crosses the threshold."
        self.play(flash_spikes(lif1_cube, num_flashes=2))
        self.wait(1)

        # Narration: "This process repeats for a second convolution and LIF layer..."
        lif2_cube = create_spike_cube((5, 10, 10), PURPLE).shift(RIGHT * 4)
        arrow2 = Arrow3D(lif1_cube.get_right(), lif2_cube.get_left(), color=WHITE, resolution=12)
        self.play(Create(arrow2), FadeIn(lif2_cube))
        
        # Narration: "...producing a final output spike train, ready for the next layer."
        self.play(flash_spikes(lif2_cube, num_flashes=3))
        self.wait(2)
