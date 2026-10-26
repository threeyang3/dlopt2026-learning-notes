import { useEffect, useState } from "react";
import { course, labNames, resource } from "./content/course";
import { Sources } from "./components/Sources";
import { GradientLab } from "./labs/GradientLab";
import { Notes } from "./components/Notes";
import { LearningContext } from "./components/LearningContext";
import { StochasticLab } from "./labs/StochasticLab";
import { LinearLab } from "./labs/LinearLab";
import { ZerothLab } from "./labs/ZerothLab";
import { DistributedLab } from "./labs/DistributedLab";
import { AttentionLab } from "./labs/AttentionLab";
import { PrecisionLab } from "./labs/PrecisionLab";
import { InferenceLab } from "./labs/InferenceLab";
import { ResourcesLab } from "./labs/ResourcesLab";
function routeAnchor(route: string) {
  const match = /^lecture\/\d+\/(.+)$/.exec(route);
  if (!match) return undefined;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return match[1];
  }
}

function useRoute() {
  const [route, setRoute] = useState(location.hash.slice(1) || "home");
  useEffect(() => {
    const listener = () => {
      const next = location.hash.slice(1) || "home";
      setRoute(next);
      if (!routeAnchor(next)) window.scrollTo(0, 0);
    };
    addEventListener("hashchange", listener);
    return () => removeEventListener("hashchange", listener);
  }, []);
  useEffect(() => {
    const anchor = routeAnchor(route);
    if (anchor) {
      document.getElementById(anchor)?.scrollIntoView({ block: "start" });
    }
  }, [route]);
  return route;
}
function stored<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
  } catch {
    return fallback;
  }
}
export function App() {
  const route = useRoute(),
    [theme, setTheme] = useState(() => stored("dlopt-theme", "light")),
    [done, setDone] = useState<number[]>(() => stored("dlopt-progress", []));
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("dlopt-theme", JSON.stringify(theme));
    } catch {}
  }, [theme]);
  function toggle(id: number) {
    const next = done.includes(id)
      ? done.filter((x) => x !== id)
      : [...done, id];
    setDone(next);
    try {
      localStorage.setItem("dlopt-progress", JSON.stringify(next));
    } catch {}
  }
  const lecture = route.startsWith("lecture/")
    ? course.find((l) => l.id === +route.split("/")[1])
    : undefined;
  return (
    <>
      <a
        className="skip"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          const main = document.getElementById("main");
          main?.focus({ preventScroll: true });
          main?.scrollIntoView({ block: "start" });
        }}
      >
        跳至主要内容
      </a>
      <header>
        <a className="brand" href="#home">
          <span className="brand-mark">∇</span>
          <span>
            DLOpt <b>2026</b>
            <small>Interactive Learning Lab</small>
          </span>
        </a>
        <nav aria-label="主导航">
          <a aria-current={route === "home" ? "page" : undefined} href="#home">
            实验首页
          </a>
          <a aria-current={route === "map" ? "page" : undefined} href="#map">
            课程地图
          </a>
          <a aria-current={route === "gd" ? "page" : undefined} href="#gd">
            梯度下降
          </a>
          <a href="#sources">资料与核验</a>
        </nav>
        <button
          aria-label="切换主题"
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
        >
          {theme === "light" ? "深色" : "浅色"}
        </button>
      </header>
      <div className="shell">
        <aside className="sidebar">
          <p>学习工作台</p>
          <a href="#map">
            14 讲知识地图 <span>{done.length}/14</span>
          </a>
          <h3>动态实验</h3>
          {Object.entries(labNames).map(([key, name]) => (
            <a
              aria-current={route === key ? "page" : undefined}
              key={key}
              href={"#" + key}
            >
              {name}
            </a>
          ))}
          <div className="sidebar-note">
            先预测，后运行。
            <br />
            让每一次参数变化
            <br />
            都有数学依据。
          </div>
        </aside>
        <main id="main" tabIndex={-1}>
          <LearningContext lab={route} />
          {route === "home" && (
            <>
              <GradientLab preview />
              <section className="home-path">
                <div>
                  <h2>从曲率出发，走到大模型训练。</h2>
                  <p>
                    14
                    讲原始笔记构成知识底座。先理解方向与下降，再观察噪声、低秩和系统成本。
                  </p>
                  <a className="button primary" href="#intro">
                    查看课程导论
                  </a>{" "}
                  <a className="button" href="#gd">
                    进入完整 GD 实验室
                  </a>
                </div>
                <ol>
                  <li>
                    <a href="#lecture/2">数学语言</a>
                    <span>梯度 · Hessian · 概率</span>
                  </li>
                  <li>
                    <a href="#gd">优化机制</a>
                    <span>GD · 动量 · 随机估计</span>
                  </li>
                  <li>
                    <a href="#lecture/8">训练与推理</a>
                    <span>Attention · 显存 · 通信</span>
                  </li>
                </ol>
              </section>
              <CourseMap done={done} toggle={toggle} />
            </>
          )}
          {route === "map" && (
            <>
              <div className="lab-intro">
                <p>课程路径</p>
                <h1>14 讲，一条可追溯的学习路线。</h1>
                <p>
                  进度仅保存在本机。掌握程度由你标记，不将浏览次数当作学会。
                </p>
              </div>
              <CourseMap done={done} toggle={toggle} />
            </>
          )}
          {route === "intro" && <CourseIntroduction />}
          {route === "gd" && (
            <>
              <GradientLab />
              <Sources ids={[5]} />
            </>
          )}
          {lecture && (
            <>
              <div className="lab-intro">
                <p>
                  Lecture {String(lecture.id).padStart(2, "0")} ·{" "}
                  {lecture.category}
                </p>
                <h1>{lecture.english}</h1>
                <p>{lecture.evidence}</p>
                {lecture.id === 1 ? (
                  <a className="button primary" href={"#" + lecture.lab}>
                    关联实验：{labNames[lecture.lab]}
                  </a>
                ) : (
                  <a className="button primary" href={"#" + lecture.lab}>
                    打开{labNames[lecture.lab]}实验室
                  </a>
                )}{" "}
                {lecture.id === 1 && (
                  <a className="button" href="#intro">
                    课程导论与学习路线
                  </a>
                )}{" "}
                <button onClick={() => toggle(lecture.id)}>
                  {done.includes(lecture.id) ? "取消掌握标记" : "标记已掌握"}
                </button>
              </div>
              <Notes id={lecture.id} />
              <Sources ids={[lecture.id]} />
            </>
          )}
          {route === "sources" && (
            <>
              <h1>资料与核验边界</h1>
              <p>
                verified
                表示明确范围内公式、条件和数值验证均有证据；partially_verified
                表示仍有来源或内容未独立复核；unverified
                表示尚未核验。课程整讲目前统一为
                partially_verified，不能由笔记既有审查记录直接升级为本轮全面核验。
              </p>
              <p>
                课程核心按现有 COURSE_SCOPE.md
                划分。跨年份复用不等于补充；延伸论文见 REFERENCES.md。第 9 讲
                Lite / MuonM 尚待课堂材料确认。
              </p>
              <a href={resource("SOURCE_AUDIT.md")}>查看原始审校记录</a>
              <Sources ids={course.map((x) => x.id)} />
            </>
          )}
          {route === "sgd" && <StochasticLab />}
          {route === "linear" && <LinearLab />}
          {route === "subspace" && <LinearLab subspace />}
          {route === "zeroth" && <ZerothLab />}
          {route === "distributed" && <DistributedLab />}
          {route === "attention" && <AttentionLab key="attention" />}
          {route === "flash" && <AttentionLab key="flash" flash />}
          {route === "precision" && <PrecisionLab />}
          {route === "inference" && <InferenceLab />}
          {route === "resources" && <ResourcesLab />}
          {![
            "home",
            "intro",
            "map",
            "sources",
            ...Object.keys(labNames),
          ].includes(route) &&
            !lecture && (
              <section>
                <h1>未找到页面</h1>
                <a href="#map">回到课程地图</a>
              </section>
            )}

          <footer>
            DLOpt 2026 Interactive Learning Lab{" "}
            <span>本地计算 · 原始笔记可追溯 · 教学实验</span>
          </footer>
        </main>
      </div>
    </>
  );
}
function CourseIntroduction() {
  const stages = [
    {
      title: "先把问题与数学语言说清楚",
      range: "第 1–4 讲",
      ids: [1, 2, 3, 4],
      summary:
        "从训练目标、梯度与 Hessian，走到矩阵、线性变换、概率、期望和方差。",
    },
    {
      title: "理解优化算法如何更新参数",
      range: "第 5–6 讲",
      ids: [5, 6],
      summary: "比较 GD、动量与加速，再理解随机梯度、批大小、噪声和 Adam。",
    },
    {
      title: "拆开 Transformer 的计算过程",
      range: "第 7–8 讲",
      ids: [7, 8],
      summary:
        "逐步计算 Attention，并理解 FlashAttention 如何改变中间存储与 IO。",
    },
    {
      title: "理解训练与推理的效率边界",
      range: "第 9–14 讲",
      ids: [9, 10, 11, 12, 13, 14],
      summary:
        "继续学习优化器状态、低秩与零阶方法、分布式通信、数值精度和 KV Cache。",
    },
  ];

  return (
    <>
      <div className="lab-intro">
        <p>课程导论</p>
        <h1>从优化问题到高效大模型训练</h1>
        <p>
          这 14
          讲不是互不相关的主题清单，而是一条从数学基础、优化算法到模型计算与系统成本的学习主线。
        </p>
      </div>
      <section className="home-path">
        <div>
          <h2>建议的学习节奏</h2>
          <p>
            每讲先明确目标、变量与适用条件；再读原始笔记并推导关键公式；最后运行对应实验，改变一个参数，解释结果为何变化。
          </p>
          <a className="button primary" href="#lecture/1">
            从第 1 讲开始
          </a>{" "}
          <a className="button" href="#map">
            浏览完整 14 讲
          </a>
        </div>
        <div>
          <h2>每讲都按这三步走</h2>
          <ol>
            <li>
              <strong>先预测。</strong>
              <span>写下更新方向、预期趋势与成立条件。</span>
            </li>
            <li>
              <strong>再演算。</strong>
              <span>跟踪公式中的变量、维度与中间结果。</span>
            </li>
            <li>
              <strong>最后验证。</strong>
              <span>改变参数，对照计算结果并记录反例。</span>
            </li>
          </ol>
        </div>
      </section>
      <section>
        <div className="section-heading">
          <h2>课程主线</h2>
          <span>14 讲 · 4 个阶段</span>
        </div>
        <div className="intro-stages">
          {stages.map((stage) => (
            <article key={stage.title}>
              <p className="intro-stage-range">{stage.range}</p>
              <h2>{stage.title}</h2>
              <p>{stage.summary}</p>
              <div className="intro-stage-links">
                {stage.ids.map((id) => {
                  const lecture = course.find((item) => item.id === id)!;
                  return (
                    <a key={id} href={"#lecture/" + id}>
                      第 {id} 讲 · {lecture.english}
                    </a>
                  );
                })}
              </div>
            </article>
          ))}
        </div>
      </section>
      <section className="notice">
        <strong>笔记与动态实验的关系</strong>
        <p>
          原始 Markdown
          是课程知识底座；动态实验只对部分核心机制建立可运行的教学模型，不代表整讲内容或真实
          GPU / 大模型实现都已复现。
        </p>
        <p>
          第 1 讲关联的 Attention 实验用于演示模型计算流程，不是第 1
          讲的完整实验，也不包含完整 Transformer 的所有组件。
        </p>
      </section>
    </>
  );
}

function CourseMap({
  done,
  toggle,
}: {
  done: number[];
  toggle: (id: number) => void;
}) {
  return (
    <section>
      <div className="section-heading">
        <h2>课程知识地图</h2>
        <span>已掌握 {done.length} / 14</span>
      </div>
      <div className="course-map">
        {course.map((l) => (
          <article key={l.id}>
            <span className="lecture-number">
              {String(l.id).padStart(2, "0")}
            </span>
            <div>
              <a className="course-title" href={"#lecture/" + l.id}>
                {l.english}
              </a>
              <p>{l.goals.replace(/\[\[.*?\]\]/g, "相关课程")}</p>
              <div className="topic-line">
                {l.topics.slice(0, 4).join(" / ")}
              </div>
              <p className="muted">
                先修：
                {l.prerequisites.length
                  ? l.prerequisites.map((id) => (
                      <a key={id} href={"#lecture/" + id}>
                        第 {id} 讲{" "}
                      </a>
                    ))
                  : "无"}{" "}
                · 课程核心 · partially_verified
              </p>
              <a href={"#" + l.lab}>
                {l.id === 1 ? "关联实验：" : "实验："}
                {labNames[l.lab]}
              </a>{" "}
              <a href={resource(l.note)}>原始 Markdown</a>
              <p className="muted">{l.id === 9 ? l.scope : ""}</p>
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={done.includes(l.id)}
                onChange={() => toggle(l.id)}
              />
              已掌握
            </label>
          </article>
        ))}
      </div>
    </section>
  );
}
