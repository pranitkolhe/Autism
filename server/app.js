import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import axios from 'axios';

dotenv.config();

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));


let sensorDataStore = [];
const MAX_DATA_POINTS = 100;

app.post("/sensor", (req, res) => {
  const data = { ...req.body, time: Date.now() };
  sensorDataStore.push(data);
  if (sensorDataStore.length > MAX_DATA_POINTS) {
    sensorDataStore.shift(); 
  }
  console.log("New Data:", data);
  res.json({ success: true });
});


app.get("/api/data", (req, res) => {
  res.json(sensorDataStore);
});

app.post("/predict", async (req, res) => {
    try {
        const formData = req.body;
        console.log("Forwarding data to ML model:", formData);


        const mlApiResponse = await axios.post('http://127.0.0.1:5000/predict_ml', formData);

        console.log("Received prediction report from ML model.");

        res.json(mlApiResponse.data);

    } catch (error) {
        console.error("Error calling ML API:", error.message);
        res.status(500).json({ error: "Failed to get prediction from ML service." });
    }
});


app.get("/", (req, res) => {

  res.render("pages/index");
});


const PORT = 4000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});