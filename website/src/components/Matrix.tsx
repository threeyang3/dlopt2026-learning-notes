import { fmt } from "./Controls";
export function MatrixView({ data, name }: { data: number[][]; name: string }) {
  return (
    <div>
      <h3>
        {name}{" "}
        <small>
          ({data.length}×{data[0].length})
        </small>
      </h3>
      <div
        className="matrix"
        role="table"
        aria-label={name}
        style={{ gridTemplateColumns: `repeat(${data[0].length},1fr)` }}
      >
        {data.flatMap((row, i) =>
          row.map((x, j) => (
            <span role="cell" key={i + "," + j} title={`${i + 1},${j + 1}`}>
              {fmt(x)}
            </span>
          )),
        )}
      </div>
    </div>
  );
}
