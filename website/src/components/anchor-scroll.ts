/** Align a rendered heading after load/font layout, without a smooth-scroll race. */
export function scheduleAnchorScroll(id: string, isCurrent: () => boolean) {
  const target = document.getElementById(id);
  if (!target) return () => {};
  let active = true;
  let firstFrame = 0;
  let secondFrame = 0;
  let resolveLoad: (() => void) | undefined;
  const onLoad = () => resolveLoad?.();
  const loaded =
    document.readyState === "complete"
      ? Promise.resolve()
      : new Promise<void>((resolve) => {
          resolveLoad = resolve;
          window.addEventListener("load", onLoad, { once: true });
        });
  const cancel = () => {
    active = false;
    window.cancelAnimationFrame(firstFrame);
    window.cancelAnimationFrame(secondFrame);
    window.removeEventListener("load", onLoad);
    window.removeEventListener("wheel", onInput);
    window.removeEventListener("touchmove", onInput);
    window.removeEventListener("keydown", onInput);
  };
  const onInput = (event: Event) => {
    if (
      event.type !== "keydown" ||
      [
        "ArrowUp",
        "ArrowDown",
        "PageUp",
        "PageDown",
        "Home",
        "End",
        " ",
      ].includes((event as KeyboardEvent).key)
    )
      cancel();
  };
  window.addEventListener("wheel", onInput, { passive: true });
  window.addEventListener("touchmove", onInput, { passive: true });
  window.addEventListener("keydown", onInput);
  // Force layout before reading fonts.ready so newly rendered KaTeX fonts are requested.
  target.getBoundingClientRect();
  const fonts = document.fonts?.ready ?? Promise.resolve();
  void Promise.all([loaded, fonts]).then(() => {
    if (!active) return;
    firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        if (active && isCurrent() && target.isConnected)
          target.scrollIntoView({ block: "start", behavior: "instant" });
        cancel();
      });
    });
  });
  return cancel;
}
