import { afterEach, expect, it, vi } from "vitest";
import { scheduleAnchorScroll } from "../components/anchor-scroll";

function setup() {
  let fontReady!: () => void;
  const fonts = new Promise<void>((resolve) => {
    fontReady = resolve;
  });
  const listeners = new Map<string, EventListener>();
  const frames = new Map<number, FrameRequestCallback>();
  let next = 1;
  const target = {
    getBoundingClientRect: vi.fn(),
    scrollIntoView: vi.fn(),
    isConnected: true,
  };
  vi.stubGlobal("document", {
    readyState: "loading",
    fonts: { ready: fonts },
    getElementById: () => target,
  });
  vi.stubGlobal("window", {
    addEventListener: (name: string, fn: EventListener) =>
      listeners.set(name, fn),
    removeEventListener: (name: string) => listeners.delete(name),
    requestAnimationFrame: (fn: FrameRequestCallback) => {
      const id = next++;
      frames.set(id, fn);
      return id;
    },
    cancelAnimationFrame: (id: number) => frames.delete(id),
  });
  const flush = async () => {
    await Promise.resolve();
    await Promise.resolve();
    while (frames.size) {
      const [id, fn] = frames.entries().next().value!;
      frames.delete(id);
      fn(0);
    }
  };
  return {
    target,
    fontReady,
    listeners,
    frames,
    flush,
    load: () => listeners.get("load")?.(new Event("load")),
  };
}
afterEach(() => vi.unstubAllGlobals());

it("load 与字体都完成后才以 instant 对齐，不受 CSS smooth 影响", async () => {
  const s = setup();
  scheduleAnchorScroll("heading", () => true);
  expect(s.target.getBoundingClientRect).toHaveBeenCalledOnce();
  s.load();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
  s.fontReady();
  await s.flush();
  expect(s.target.scrollIntoView).toHaveBeenCalledExactlyOnceWith({
    block: "start",
    behavior: "instant",
  });
  expect(s.listeners.size).toBe(0);
});
it("字体先完成仍等待页面加载，清理后不残留定位任务", async () => {
  const s = setup();
  const cancel = scheduleAnchorScroll("heading", () => true);
  s.fontReady();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
  s.load();
  cancel();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
  expect(s.listeners.size).toBe(0);
  expect(s.frames.size).toBe(0);
});
it("旧路由完成异步等待后不能滚动新页面", async () => {
  const s = setup();
  scheduleAnchorScroll("heading", () => false);
  s.load();
  s.fontReady();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
});
it("用户主动滚动后不再把页面拉回锚点", async () => {
  const s = setup();
  scheduleAnchorScroll("heading", () => true);
  s.listeners.get("wheel")?.(new Event("wheel"));
  s.load();
  s.fontReady();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
  expect(s.listeners.size).toBe(0);
});
it("已脱离文档的标题不再触发滚动", async () => {
  const s = setup();
  scheduleAnchorScroll("heading", () => true);
  s.target.isConnected = false;
  s.load();
  s.fontReady();
  await s.flush();
  expect(s.target.scrollIntoView).not.toHaveBeenCalled();
});
