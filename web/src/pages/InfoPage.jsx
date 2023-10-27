import { VictoryChart, VictoryArea, VictoryTheme } from "victory";

export default function InfoPage() {
  const diseases = [
    "DR",
    "ARMD",
    "MH",
    "DN",
    "MYA",
    "BRVO",
    "TSLN",
    "ERM",
    "LS",
    "MS",
    "CSR",
    "ODC",
    "CRVO",
    "TV",
    "AH",
    "ODP",
    "ODE",
    "ST",
    "AION",
    "PT",
    "RT",
    "RS",
    "CRS",
    "EDN",
    "RPEC",
    "MHL",
    "RP",
    "CWS",
    "CB",
    "ODPM",
    "PRH",
    "MNF",
    "HR",
    "CRAO",
    "TD",
    "CME",
    "PTCR",
    "CF",
    "VH",
    "MCA",
    "VS",
    "BRAO",
    "PLQ",
    "HPED",
    "CL",
  ];

  const specificDiseases = [
    { label: "DR", accuracy: 0.1, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "ARMD", accuracy: 0.4, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "MH", accuracy: 0.2, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "DN", accuracy: 0.5, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "MYA", accuracy: 0.2, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "BRVO", accuracy: 0.3, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "TSLN", accuracy: 0.3, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "LS", accuracy: 0.9, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "CSR", accuracy: 0.4, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "ODC", accuracy: 0.3, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "CRVO", accuracy: 0.4, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "ODP", accuracy: 0.5, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "ODE", accuracy: 0.1, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "RS", accuracy: 0.2, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "CRS", accuracy: 0.3, recall: 0.5, f1: 0.5, precision: 0.5 },
    { label: "RPEC", accuracy: 0.5, recall: 0.5, f1: 0.5, precision: 0.5 },
  ];

  return (
    <div className="text-center w-screen p-8 px-16 flex flex-col gap-2">
      <a className="text-5xl font-bold font-logo text-cyan-400" href="/">
        TruthEyes
      </a>
      <div className="text-2xl">A tool to detect eyes diseases</div>
      <div className="text-left flex flex-col gap-2">
        <div className="text-lg font-logo">Unveiling Trutheyes</div>
        <div className="mb-4">
          Shaping a Future of Early Eye Cancer Detection At Trutheyes, our
          vision transcends technology; it&apos;s about safeguarding your health
          and well-being. We embark on a mission to address a pressing
          need—early detection of eye cancer. This devastating disease demands
          attention, and we&apos;re here to fill a crucial gap.
        </div>

        <div className="text-lg font-logo">Understanding the Need</div>
        <div className="mb-4">
          Early Detection Is Vital Eye cancer may be relatively rare, but its
          consequences are profound. Early diagnosis can mean the difference
          between timely intervention and potentially life-altering outcomes.
          Trutheyes recognizes the urgency of providing a solution that empowers
          individuals to take control of their ocular health.
        </div>
        <div className="text-lg font-logo">Our Commitment</div>
        <div className="mb-4">
          Empowering Patients and Healthcare Professionals. Our unwavering
          commitment is to bring advanced diagnostics to your fingertips.
          Trutheyes aspires to be the bridge between individuals and medical
          practitioners. We&apos;re making it easier for healthcare
          professionals to make accurate and timely diagnoses, and for patients
          to access the care they need.
        </div>
        <div className="text-lg font-logo">The Heart of Trutheyes</div>
        <div className="mb-4">
          Transparency and Trust At Trutheyes, transparency is our guiding
          principle. We believe that understanding the &quot;why&quot; behind a
          medical diagnosis is essential. Our commitment to trust is embodied in
          the AI models we&apos;ve created, which provide insights into the
          decision-making process. Your Role in the Vision You are at the center
          of our vision. Trutheyes seeks to bring you closer to your ocular
          health. Together, we can revolutionize early eye cancer detection and
          make it more accessible than ever before.
          <br />
          <br />
          Actualy, we are able to detect:&#20;
          {diseases.map((disease, i) => (
            <span
              key={i}
              className={
                specificDiseases.map((x) => x.label).includes(disease)
                  ? "font-medium"
                  : ""
              }
            >
              {disease}
              {i != diseases.length - 1 ? ", " : ""}
            </span>
          ))}
          <div className="grid grid-cols-4 w-full">
            <div className="flex flex-col">
              <VictoryChart
                polar
                theme={VictoryTheme.material}
                domain={{ y: [0, 1] }}
                animate={{
                  duration: 2000,
                  onLoad: { duration: 1000 },
                }}
              >
                <VictoryArea
                  data={specificDiseases}
                  y="accuracy"
                  style={{
                    data: {
                      fill: "#00B0FF",
                    },
                  }}
                />
              </VictoryChart>
              <div className="text-lg text-center">Accuracy</div>
            </div>
            <div className="flex flex-col">
              <VictoryChart
                polar
                theme={VictoryTheme.material}
                domain={{ y: [0, 1] }}
                animate={{
                  duration: 2000,
                  onLoad: { duration: 1000 },
                }}
              >
                <VictoryArea
                  data={specificDiseases}
                  y="recall"
                  style={{
                    data: {
                      fill: "#00B0FF",
                    },
                  }}
                />
              </VictoryChart>
              <div className="text-lg text-center">Recall</div>
            </div>
            <div className="flex flex-col">
              <VictoryChart
                polar
                theme={VictoryTheme.material}
                domain={{ y: [0, 1] }}
                animate={{
                  duration: 2000,
                  onLoad: { duration: 1000 },
                }}
              >
                <VictoryArea
                  data={specificDiseases}
                  y="precision"
                  style={{
                    data: {
                      fill: "#00B0FF",
                    },
                  }}
                />
              </VictoryChart>
              <div className="text-lg text-center">Recall</div>
            </div>
            <div className="flex flex-col">
              <VictoryChart
                polar
                theme={VictoryTheme.material}
                domain={{ y: [0, 1] }}
                animate={{
                  duration: 2000,
                  onLoad: { duration: 1000 },
                }}
              >
                <VictoryArea
                  data={specificDiseases}
                  y="f1"
                  style={{
                    data: {
                      fill: "#00B0FF",
                    },
                  }}
                />
              </VictoryChart>
              <div className="text-lg text-center">F1 Score</div>
            </div>
          </div>
        </div>
        <div className="text-lg font-logo">Join Us on the Journey</div>
        <div className="mb-4">
          Trutheyes is not just a project; it&apos;s a commitment to your
          well-being. We invite you to join us in shaping a future where early
          detection is within everyone&apos;s reach. Your ocular health matters,
          and Trutheyes is here to make a difference.
        </div>
      </div>
    </div>
  );
}
