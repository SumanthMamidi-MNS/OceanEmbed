import { Icon, IconButton } from "@/components/ui/primitives";
import type { LinkedView } from "@/state/linkedView";
import { useZoom } from "@/state/useLinkedView";

/** Zoom buttons for a group of linked maps (the keyboard and pointer equivalents live on the maps). */
export function ZoomControls({ link }: { link: LinkedView }) {
  const k = useZoom(link);
  return (
    <div className="zoomctl" role="group" aria-label="Map zoom (all linked maps)">
      <IconButton label="Zoom out" onClick={() => link.zoomBy(1 / 1.6)} disabled={k <= 1}>
        <Icon name="minus" />
      </IconButton>
      <span className="zoomctl__k num" aria-live="polite">
        {k.toFixed(k < 10 ? 1 : 0)}×
      </span>
      <IconButton label="Zoom in" onClick={() => link.zoomBy(1.6)}>
        <Icon name="plus" />
      </IconButton>
      <IconButton label="Reset view" onClick={() => link.reset()} disabled={k <= 1}>
        <Icon name="reset" />
      </IconButton>
    </div>
  );
}
