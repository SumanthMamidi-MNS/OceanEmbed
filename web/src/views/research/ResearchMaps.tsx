/**
 * Research · methods on the map: the Explorer with every estimate of the run. It opens in Compare
 * (every estimate beside GLORYS on shared scales) and offers the estimate selector; the product's
 * Explorer shows the product alone.
 */
import { ExplorerBody } from "../explorer/Explorer";
import { ResearchHead } from "./ResearchHead";

export default function ResearchMaps() {
  return (
    <>
      <ResearchHead />
      <ExplorerBody />
    </>
  );
}
