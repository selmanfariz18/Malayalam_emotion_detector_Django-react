import React, { useState } from "react";

const malayalamKeys = [
  "അ", "ആ", "ഇ", "ഈ", "ഉ", "ഊ", "എ", "ഏ", "ഐ", "ഒ", "ഓ", "ഔ",
  "ക", "ഖ", "ഗ", "ഘ", "ങ", "ച", "ഛ", "ജ", "ഝ", "ഞ",
  "ട", "ഠ", "ഡ", "ഢ", "ണ", "ത", "ഥ", "ദ", "ധ", "ന",
  "പ", "ഫ", "ബ", "ഭ", "മ", "യ", "ര", "ല", "വ", "ശ",
  "ഷ", "സ", "ഹ", "ള", "ഴ", "റ", "ം", "ഃ", "ൠ", "ൡ"
];

function VirtualKeyboard({ onKeyClick }) {
  const [position, setPosition] = useState({ x: 100, y: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [size, setSize] = useState({ width: 500, height: 300 });
  const [isResizing, setIsResizing] = useState(false);
  const [startResizePos, setStartResizePos] = useState({ x: 0, y: 0 });

  // Dragging Logic
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setStartResizePos({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - startResizePos.x,
        y: e.clientY - startResizePos.y,
      });
    } else if (isResizing) {
      const deltaX = e.clientX - startResizePos.x;
      const deltaY = e.clientY - startResizePos.y;
      setSize((prevSize) => ({
        width: Math.max(300, prevSize.width + deltaX),
        height: Math.max(200, prevSize.height + deltaY),
      }));
      setStartResizePos({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };

  // Resizing Logic
  const handleResizeMouseDown = (e) => {
    e.stopPropagation(); // Prevent conflict with drag logic
    setIsResizing(true);
    setStartResizePos({ x: e.clientX, y: e.clientY });
  };

  return (
    <div
      className="virtual-keyboard-wrapper"
      style={{
        position: "absolute",
        top: `${position.y}px`,
        left: `${position.x}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
        cursor: isDragging ? "grabbing" : "grab",
        zIndex: 1000,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Background Layout */}
      <div className="keyboard-background">
        <div
          className="virtual-keyboard"
          style={{
            gridTemplateColumns: `repeat(auto-fit, minmax(${size.width / 10}px, 1fr))`,
          }}
        >
          {malayalamKeys.map((key, index) => (
            <button
              key={index}
              className="key-button"
              onClick={() => onKeyClick(key)}
            >
              {key}
            </button>
          ))}
        </div>
        {/* Resize Button */}
        <div
          className="resize-button"
          onMouseDown={handleResizeMouseDown}
        ></div>
      </div>
    </div>
  );
}

export default VirtualKeyboard;
