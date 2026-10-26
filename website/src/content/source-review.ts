// PDF 页序从 1 起。只收录本轮实际读取的页，而非照抄原审校结论。
export const sourceReview: Record<
  number,
  { path: string; page: number; topic: string; method: string }[]
> = {
  5: [
    {
      path: "materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf",
      page: 11,
      topic: "GD 更新",
      method: "PDF 文本核对",
    },
    {
      path: "materials/Lecture-05_Gradient-Descent/01_convexity-gradient-descent_slides.pdf",
      page: 14,
      topic: "光滑下降界；驻点处不能声称严格下降",
      method: "PDF 文本与原页图像核对",
    },
    {
      path: "materials/Lecture-05_Gradient-Descent/03_accelerated-gradient-descent_slides.pdf",
      page: 5,
      topic: "Heavy-ball 位移递推；本页速度变量为等价缩放",
      method: "PDF 文本核对",
    },
    {
      path: "materials/Lecture-05_Gradient-Descent/03_accelerated-gradient-descent_slides.pdf",
      page: 19,
      topic: "NAG 外推点评估",
      method: "PDF 文本核对",
    },
    {
      path: "materials/Lecture-05_Gradient-Descent/03_accelerated-gradient-descent_slides.pdf",
      page: 23,
      topic: "强凸 NAG 推荐参数与函数间隙速率",
      method: "PDF 文本与原页图像核对",
    },
  ],
  6: [
    {
      path: "materials/Lecture-06_Stochastic-Gradient-Descent/05_adaptive-sgd_slides.pdf",
      page: 18,
      topic: "Adam 一阶与二阶矩；课件省略偏差修正，网站按原笔记补全",
      method: "PDF 文本核对",
    },
  ],
  8: [
    {
      path: "materials/Lecture-08_Transformer-and-LLM-II/03_flash-attention_slides.pdf",
      page: 21,
      topic: "Online softmax 的最大值与归一化和合并",
      method: "PDF 文本与原页图像核对",
    },
  ],
};
