import { useEffect, useState } from "react";
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="slider">
      <span>
        {label}
        <output>{Number(value.toPrecision(5))}</output>
      </span>
      <input
        type="range"
        aria-label={label}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(+e.target.value)}
      />
    </label>
  );
}
export function usePlayback(max: number) {
  const [index, setIndex] = useState(0),
    [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () => setIndex((k) => Math.min(k + 1, max)),
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 900 : 180,
    );
    return () => clearInterval(timer);
  }, [playing, max]);
  useEffect(() => {
    if (index >= max) setPlaying(false);
  }, [index, max]);
  const reset = () => {
    setPlaying(false);
    setIndex(0);
  };
  return { index, playing, setIndex, setPlaying, reset };
}
export function Playback({
  player,
  max,
}: {
  player: ReturnType<typeof usePlayback>;
  max: number;
}) {
  return (
    <div className="playback">
      <button
        className="primary"
        disabled={player.index >= max}
        onClick={() => player.setPlaying(!player.playing)}
      >
        {player.playing ? "暂停" : "播放"}
      </button>
      <button
        aria-label="后退一步"
        disabled={player.index === 0}
        onClick={() => {
          player.setPlaying(false);
          player.setIndex(player.index - 1);
        }}
      >
        后退
      </button>
      <button
        aria-label="前进一步"
        disabled={player.index >= max}
        onClick={() => {
          player.setPlaying(false);
          player.setIndex(player.index + 1);
        }}
      >
        单步
      </button>
      <button onClick={player.reset}>重置</button>
      <label className="scrub">
        迭代 <output data-testid="iteration">{player.index}</output>
        <input
          aria-label="迭代位置"
          type="range"
          min="0"
          max={max}
          value={player.index}
          onChange={(e) => {
            player.setPlaying(false);
            player.setIndex(+e.target.value);
          }}
        />
      </label>
    </div>
  );
}
export const fmt = (x: number) =>
  !Number.isFinite(x)
    ? String(x)
    : Math.abs(x) > 1e4 || (Math.abs(x) < 1e-3 && x !== 0)
      ? x.toExponential(3)
      : x.toFixed(4);
export const vector = (x: number[]) => "[" + x.map(fmt).join(", ") + "]";
