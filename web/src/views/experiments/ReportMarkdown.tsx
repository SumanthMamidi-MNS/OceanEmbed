/** The generated report.md rendered to React elements (no raw HTML). Loaded only when it is opened. */
import Markdown from "markdown-to-jsx";

function ReportImage(props: { src?: string; alt?: string; resolve: (src: string) => string }) {
  return <img src={props.resolve(props.src ?? "")} alt={props.alt ?? "Report figure"} loading="lazy" decoding="async" />;
}

function ScrollTable(props: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="tablewrap">
      <table {...props} className="table table--dense" />
    </div>
  );
}

export default function ReportMarkdown({ markdown, resolve }: { markdown: string; resolve: (src: string) => string }) {
  return (
    <div className="markdown">
      <Markdown
        options={{
          disableParsingRawHTML: true,
          overrides: {
            img: { component: ReportImage, props: { resolve } },
            a: { props: { rel: "noreferrer" } },
            table: { component: ScrollTable },
          },
        }}
      >
        {markdown}
      </Markdown>
    </div>
  );
}
