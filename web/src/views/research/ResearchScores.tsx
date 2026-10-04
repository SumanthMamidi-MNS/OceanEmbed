/**
 * Research · every score: the Accuracy page with every method of the run (ridge regression, the
 * per-pixel network, the ablation) beside the model, and the estimate selector. The method scope
 * comes from the route (state/runContext), so the sections are the same components.
 */
import { AccuracyBody } from "../accuracy/Accuracy";
import { ResearchHead } from "./ResearchHead";

export default function ResearchScores() {
  return (
    <>
      <ResearchHead />
      <AccuracyBody />
    </>
  );
}
