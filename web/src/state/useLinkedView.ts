import { useEffect, useMemo, useSyncExternalStore } from "react";
import type { GridGeom } from "@/lib/geo";
import { LinkedView } from "./linkedView";

/** One shared camera + cursor for a group of maps over the same domain. */
export function useLinkedView(geom: GridGeom): LinkedView {
  const { lon0, lon1, lat0, lat1 } = geom;
  const link = useMemo(() => new LinkedView({ lon0, lon1, lat0, lat1 }), []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    link.setDomain({ lon0, lon1, lat0, lat1 });
  }, [link, lon0, lon1, lat0, lat1]);
  return link;
}

/** Current zoom factor of a linked view (re-renders only when it changes). */
export function useZoom(link: LinkedView): number {
  return useSyncExternalStore(link.subscribeView, () => link.k);
}
