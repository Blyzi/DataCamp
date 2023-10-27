import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import InfoPage from "./pages/InfoPage";
import ResultPage from "./pages/ResultPage";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}></Route>
        <Route path="/info" element={<InfoPage />}></Route>
        <Route path="/results" element={<ResultPage />}></Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
