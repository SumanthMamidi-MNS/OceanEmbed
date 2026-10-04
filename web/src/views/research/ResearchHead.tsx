/**
 * The band at the top of every Research page: what this area is, its sub-pages and the way back
 * to the product. The Research area documents how the model was chosen; a user of the product
 * does not need it, so it is linked from the colophon and from Data & downloads only.
 */
import { ViewLink } from "@/components/shell/Shell";
import { Icon } from "@/components/ui/primitives";
import { useUrlState } from "@/state/router";
import { RESEARCH_VIEWS } from "@/state/url";

export function ResearchHead() {
  const [url] = useUrlState();
  return (
    <div className="researchhead">
      <div className="researchhead__row">
        <p className="researchhead__what">
          <span className="overline">Research</span>
          <span>
            How the model was chosen and how we know: the methods compared, the ablations, the training and the embedding. The research
            notes are in the repository under <code>docs/research/</code>, the result tables under <code>results/</code>.
          </span>
        </p>
        <ViewLink view="overview" className="btn btn--quiet researchhead__back">
          Back to the product <Icon name="arrow" />
        </ViewLink>
      </div>
      <nav className="tabs" aria-label="Research pages">
        {RESEARCH_VIEWS.map((v) => (
          <ViewLink key={v.key} view={v.key} className="tabs__link" current={url.view === v.key}>
            {v.label}
          </ViewLink>
        ))}
      </nav>
    </div>
  );
}
