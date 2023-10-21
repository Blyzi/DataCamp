import { useState } from "react";
import { FileUploader } from "react-drag-drop-files";

export default function HomePage() {
  const [uploadedImage, setUploadedImage] = useState(null);

  return (
    <div className="h-screen w-screen flex flex-col justify-center items-center gap-8">
      <div className="flex flex-col justify-center items-center gap-2">
        <div className="text-5xl">
          Welcome to <span className="font-logo text-cyan-400"> TruthEyes</span>
        </div>
        <div className="text-2xl"> Let&apos;s get started! </div>
      </div>

      <div className="h-1/3 w-1/3 flex flex-col items-center gap-2">
        {uploadedImage == null ? (
          <FileUploader
            handleChange={(file) => {
              console.log(file);
              setUploadedImage(file);
            }}
            multiple={false}
            types={["png", "jpeg"]}
            classes="border-cyan-400 border-dashed border-4 h-full w-full rounded-xl flex items-center justify-center"
          >
            <div className="animate-bounce">Drop your radio here</div>
          </FileUploader>
        ) : (
          <div className="flex flex-col justify-center items-center gap-4 h-full w-full">
            <img
              src={URL.createObjectURL(uploadedImage)}
              className="object-contain w-full overflow-hidden rounded-lg bg-black"
            ></img>

            <div className="flex flex-col gap-2 justify-center items-center w-full">
              <button
                onClick={() => {
                  setUploadedImage(null);
                }}
                className="bg-cyan-400 rounded-xl text-white w-full p-2"
              >
                Send for analysis
              </button>
            </div>
          </div>
        )}
        <a className="text-xs text-gray-400" href="/info">
          Information about the analysis
        </a>
      </div>
    </div>
  );
}
