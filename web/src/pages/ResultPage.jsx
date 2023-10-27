import { useLocation } from "react-router-dom";
import { VictoryChart, VictoryBar, VictoryAxis } from "victory";
import { useNavigate } from "react-router-dom";
import { useEffect } from "react";

export default function ResultPage() {
  const { state } = useLocation();

  const navigate = useNavigate();

  const data = state?.predictions;

  useEffect(() => {
    if (state == null) {
      navigate("/");
    }
  }, [state, navigate]);

  const threshold = [
    {
      label: "Disease_Risk",
      score: 0.45,
      description: (
        <div>
          <span className="bold">Disease_Risk (Risk of Disease):</span> This
          category is not a specific disease but rather indicates the potential
          risk of having a disease or abnormality in the eye when identified in
          a retinal image. It serves as a general alert for healthcare
          professionals to further investigate the image for specific
          conditions.
        </div>
      ),
    },
    {
      label: "DR",
      score: 0.35,
      description: (
        <div>
          <span className="bold">DR (Diabetic Retinopathy):</span> Diabetic
          retinopathy is a condition that affects people with diabetes. It is
          characterized by the appearance of small red spots (microaneurysms),
          tiny bleeding points, yellowish deposits (hard exudates), or fluffy
          white patches (cotton wool spots) in the retina. These signs may
          indicate damage to blood vessels in the eye due to diabetes.
        </div>
      ),
    },
    {
      label: "ARMD",
      score: 0.4,
      description: (
        <div>
          <span className="bold">ARMD (Age-Related Macular Degeneration):</span>
          Age-related macular degeneration is a condition that often occurs in
          older individuals. It leads to central vision loss and is marked by
          the presence of yellow deposits (drusen) in the macular region,
          geographic atrophy (thinning of the retina), and sometimes the growth
          of abnormal blood vessels.
        </div>
      ),
    },
    {
      label: "MH",
      score: 0.41,
      description: (
        <div>
          <span className="bold">MH (Media Haze):</span> Media haze is a general
          term for cloudiness or haziness in the eye that can be caused by
          various factors like cataracts, corneal edema, or other eye
          conditions. It can make the retinal image appear less clear due to
          obstructions or distortions.
        </div>
      ),
    },
    {
      label: "DN",
      score: 0.39,
      description: (
        <div>
          <span className="bold">DN (Drusens):</span> Drusens are small yellow
          or white deposits that can accumulate in the retina. They are often a
          sign of aging but can also be associated with eye diseases like
          age-related macular degeneration. They may appear as small, round, and
          bright spots in the retina.
        </div>
      ),
    },
    {
      label: "MYA",
      score: 0.32,
      description: (
        <div>
          <span className="bold">MYA (Myopia):</span> Myopia, commonly known as
          nearsightedness, is a condition where distant objects appear blurry.
          In retinal images, it may be associated with changes in the choroid,
          sclera, and retinal pigment epithelium (RPE).
        </div>
      ),
    },
    {
      label: "BRVO",
      score: 0.4,
      description: (
        <div>
          <span className="bold">BRVO (Branch Retinal Vein Occlusion):</span>{" "}
          Branch retinal vein occlusion is a condition in which one or more
          branches of the central retinal vein are blocked. This can lead to
          visual disturbances, such as dot and blot hemorrhages, flame-shaped
          hemorrhages, and swelling in the retina.
        </div>
      ),
    },
    {
      label: "TSLN",
      score: 0.37,
      description: (
        <div>
          <span className="bold">TSLN (Tessellation):</span> Tessellation refers
          to a pattern of reduced pigment density in the retina, making the
          choroidal vessels more visible. This condition can occur with aging
          and, in some cases, may be linked to myopia.
        </div>
      ),
    },

    {
      label: "ODC",
      score: 0.36,
      description: (
        <div>
          <span className="bold">ODC (Optic Disc Cupping):</span> Optic disc
          cupping is characterized by a visibly excavated or thinned appearance
          of the optic disc. It can be a sign of glaucoma or other eye
          conditions and may result in changes in the appearance of the optic
          nerve head.
        </div>
      ),
    },

    {
      label: "ODP",
      score: 0.38,
      description: (
        <div>
          <span className="bold">ODP (Optic Disc Pallor):</span> Optic disc
          pallor refers to a pale or yellowish discoloration of the optic disc.
          It can indicate the loss of nerve fibers and may affect vision. The
          extent of discoloration is associated with the degree of visual
          impairment.
        </div>
      ),
    },
    {
      label: "ODE",
      score: 0.4,
      description: (
        <div>
          <span className="bold">ODE (Optic Disc Edema):</span> Optic disc edema
          involves swelling or inflammation of the optic disc. This can result
          from various causes and may be associated with changes in the
          appearance of the optic nerve head, often presenting as a swollen or
          choked disc.
        </div>
      ),
    },
  ].filter((x) => data.map((x) => x.label).includes(x.label));

  return (
    <div className="min-h-screen w-screen flex flex-col justify-center items-center gap-8">
      <div className="flex flex-col justify-center items-center gap-2">
        <div className="text-5xl">Here there are !</div>
        <div className="text-2xl">
          Your result from{" "}
          <a className="font-logo text-cyan-400" href="/">
            TruthEyes
          </a>
        </div>
      </div>
      <div className="h-1/3 w-1/2">
        <VictoryChart
          style={{ parent: { minWidth: "100%" } }}
          animate={{
            duration: 2000,
            onLoad: { duration: 1000 },
          }}
          domain={{ y: [0, 1] }}
        >
          <VictoryBar
            data={data}
            y="score"
            x="label"
            style={{ labels: { fontSize: 0 } }}
          />
          <VictoryBar
            data={threshold.map(({ label, score }) => ({
              label,
              score,
            }))}
            x="label"
            y="score"
            style={{
              data: {
                fill: "transparent",
                stroke: "cyan",
                strokeWidth: 2,
              },
              labels: { fontSize: 0 },
            }}
          />

          <VictoryAxis
            style={{ tickLabels: { fontSize: 12 } }}
            categories={{ x: threshold.map((x) => x.label) }}
          />
        </VictoryChart>
      </div>
      <ul className="list-disc w-2/3">
        {threshold
          .filter((x) => x.score < data.find((y) => x.label == y.label).score)
          .map((x) => (
            <li key={x.label}>{x.description}</li>
          ))}
      </ul>

      <div className=" flex flex-col items-center gap-2">
        <a className="text-xs text-gray-400" href="/info">
          Information about the analysis
        </a>
        <div className="flex flex-col items-center gap-2">
          <a
            className="border-solid border-2 border-cyan-300 px-2 py-3 rounded-lg"
            href="/"
          >
            Reupload an image
          </a>
        </div>
      </div>
    </div>
  );
}
