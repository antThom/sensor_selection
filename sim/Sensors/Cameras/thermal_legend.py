"""IR palette legend displayed only while an infrared camera is active."""

from __future__ import annotations

import numpy as np
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectFrame
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


class ThermalLegend:
    """Draw a compact live kelvin/Celsius scale at the right of the window."""

    SEGMENT_COUNT = 48

    def __init__(self, world, camera):
        self.world = world
        self.camera = camera
        self.root = world.aspect2d.attachNewNode(f"{camera.model_number}-thermal-legend")
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        self.root.setBin("fixed", 100)
        self._last_range = None
        self._last_palette = None

        aspect = float(world.getAspectRatio())
        self.bar_x = aspect - 0.24
        self.bar_bottom = -0.60
        self.bar_top = 0.60
        self.segments = []
        segment_height = (self.bar_top - self.bar_bottom) / self.SEGMENT_COUNT
        for index in range(self.SEGMENT_COUNT):
            bottom = self.bar_bottom + index * segment_height
            frame = DirectFrame(
                parent=self.root,
                frameSize=(-0.045, 0.045, bottom, bottom + segment_height + 0.002),
                frameColor=(0.0, 0.0, 0.0, 1.0),
                pos=(self.bar_x, 0.0, 0.0),
                relief=DGG.FLAT,
            )
            self.segments.append(frame)

        label_x = self.bar_x - 0.075
        self.title = OnscreenText(
            parent=self.root,
            text="Estimated\nTemperature",
            pos=(self.bar_x, self.bar_top + 0.13),
            scale=0.043,
            fg=(1.0, 1.0, 1.0, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55),
            align=TextNode.ACenter,
            mayChange=False,
        )
        self.maximum_label = self._make_label(label_x, self.bar_top)
        self.middle_label = self._make_label(label_x, 0.0)
        self.minimum_label = self._make_label(label_x, self.bar_bottom)
        self.root.hide()

    def _make_label(self, x, y):
        return OnscreenText(
            parent=self.root,
            text="",
            pos=(x, y - 0.016),
            scale=0.035,
            fg=(1.0, 1.0, 1.0, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55),
            align=TextNode.ARight,
            mayChange=True,
        )

    def set_visible(self, visible):
        if visible:
            self.root.show()
        else:
            self.root.hide()

    def update(self, temperature_range_K, palette):
        values = tuple(round(float(value), 3) for value in temperature_range_K)
        palette_name = str(palette)
        if values == self._last_range and palette_name == self._last_palette:
            return
        self._last_range = values
        self._last_palette = palette_name

        normalized = np.linspace(0.0, 1.0, self.SEGMENT_COUNT)
        colors = self.camera._apply_palette(normalized, palette_name)
        for frame, color in zip(self.segments, colors):
            frame["frameColor"] = tuple(float(channel) / 255.0 for channel in color) + (
                1.0,
            )

        minimum, maximum = values
        middle = (minimum + maximum) * 0.5
        self.minimum_label.setText(self.format_temperature(minimum))
        self.middle_label.setText(self.format_temperature(middle))
        self.maximum_label.setText(self.format_temperature(maximum))

    @staticmethod
    def format_temperature(temperature_K):
        temperature_K = float(temperature_K)
        return f"{temperature_K:.1f} K\n{temperature_K - 273.15:.1f} C"
