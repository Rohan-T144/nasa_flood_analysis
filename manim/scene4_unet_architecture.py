# scene4_unet_architecture_v0_19.py
from manim import *

class SpikingUNetArchitecture(Scene):
    def construct(self):
        # --- Setup U-Net Diagram ---
        title = Text("Spiking U-Net Architecture").to_edge(UP)
        self.play(Write(title))
        
        def create_block(text, color, width, height):
            rect = Rectangle(width=width, height=height, color=color, fill_opacity=0.5)
            label = Text(text, font_size=24).move_to(rect.get_center())
            return VGroup(rect, label)

        # Encoder (Down path)
        encoder_blocks = VGroup()
        scale = .55
        widths = [3, 2.7, 2.4, 2.1] 
        widths = [x*scale for x in widths]
        heights = [1.5, 1.2, 0.9, 0.6]
        heights = [x*scale for x in heights]
        colors = [BLUE, GREEN, YELLOW, ORANGE]
        for i, (w, h, color) in enumerate(zip(widths, heights, colors)):
            block = create_block(f"Enc {i+1}", color, w, h)
            if i > 0:
                block.next_to(encoder_blocks[i-1], DOWN, buff=0.8).align_to(encoder_blocks[i-1], RIGHT)
            encoder_blocks.add(block)
        
        bottleneck = create_block("BottleNeck", RED, widths[-1], heights[-1]).next_to(encoder_blocks[-1], DOWN+RIGHT, buff=0.8)
        
        decoder_blocks = VGroup()
        for i, (w, h, color) in enumerate(zip(reversed(widths), reversed(heights), reversed(colors))):
            block = create_block(f"Dec {4-i}", color, w, h)
            if i == 0:
                block.next_to(bottleneck, UP+RIGHT, buff=0.8)
            else:
                block.next_to(decoder_blocks[i-1], UP, buff=0.8).align_to(decoder_blocks[i-1], LEFT)
            decoder_blocks.add(block)

        unet_diagram = VGroup(encoder_blocks, bottleneck, decoder_blocks).center()
        self.play(FadeIn(unet_diagram))

        arrows = VGroup()
        for i in range(len(encoder_blocks) - 1):
            arrows.add(Arrow(encoder_blocks[i].get_bottom(), encoder_blocks[i+1].get_top(), buff=0.1))
        arrows.add(Arrow(encoder_blocks[-1].get_bottom(), bottleneck.get_top(), buff=0.1))
        arrows.add(Arrow(bottleneck.get_top(), decoder_blocks[0].get_left(), buff=0.1))
        for i in range(len(decoder_blocks) - 1):
            arrows.add(Arrow(decoder_blocks[i].get_top(), decoder_blocks[i+1].get_bottom(), buff=0.1))
        
        skip_arrows = VGroup()
        for i in range(len(encoder_blocks)):
            start_point = encoder_blocks[i].get_right()
            end_point = decoder_blocks[-(i+1)].get_left()
            skip_arrows.add(DashedLine(start_point, end_point, color=GREY, buff=0.1))
        
        # UPDATED: Replaced deprecated LaggedStartMap with modern syntax
        self.play(
            LaggedStart(*[GrowArrow(arrow) for arrow in arrows]),
            LaggedStart(*[Create(arrow) for arrow in skip_arrows])
        )
        self.wait(1)

        # --- Animate Data Flow ---
        data_cube = Dot(color=PINK, radius=0.2)
        
        # Narration: "The model uses an encoder-decoder structure. In the encoder, the input is progressively downsampled to capture context."
        self.play(data_cube.animate.move_to(encoder_blocks[0].get_center()), run_time=0.5)
        for i in range(len(encoder_blocks)):
            self.play(MoveAlongPath(data_cube, Line(data_cube.get_center(), encoder_blocks[i].get_center())), run_time=0.5)
            if i < len(encoder_blocks) - 1:
                self.play(MoveAlongPath(data_cube, Line(encoder_blocks[i].get_center(), encoder_blocks[i+1].get_center())), run_time=0.5)
        
        self.play(MoveAlongPath(data_cube, Line(encoder_blocks[-1].get_center(), bottleneck.get_center())), run_time=0.5)
        self.play(MoveAlongPath(data_cube, Line(bottleneck.get_center(), decoder_blocks[0].get_center())), run_time=0.5)
        
        # Narration: "The decoder then upsamples the features to reconstruct the segmentation map."
        for i in range(len(decoder_blocks)):
            # Narration: "Crucially, skip connections pass high-resolution features from the encoder directly to the decoder."
            self.play(Indicate(skip_arrows[-(i+1)], color=WHITE), run_time=0.8)
            # Narration: "This helps preserve fine-grained spatial details, which is vital for accurate segmentation."
            if i < len(decoder_blocks) - 1:
                self.play(MoveAlongPath(data_cube, Line(decoder_blocks[i].get_center(), decoder_blocks[i+1].get_center())), run_time=0.5)

        self.wait(2)
