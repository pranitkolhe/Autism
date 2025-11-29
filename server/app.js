import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
import axios from "axios";
import db from "./db.js";

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


// ------------------- STORE SENSOR DATA -------------------
app.post("/sensor", async (req, res) => {
    const data = { ...req.body, created_at: new Date() };

    sensorDataStore.push(data);
    if (sensorDataStore.length > MAX_DATA_POINTS) sensorDataStore.shift();

    console.log("New Sensor:", data);

    // Prepare SQL values
    const bpm = Number(data.bpm) || 0;
    const bpm_status =
        bpm === 0 ? "0" : bpm > 80 ? "High" : "Normal";

    const repetitive = data.repetitive === "YES" ? "YES" : "NO";

    const sql = `
        INSERT INTO sensor_logs 
        (bpm, bpm_status, repetitive, ax, ay, az, gx, gy, gz, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    db.query(
        sql,
        [
            bpm,
            bpm_status,
            repetitive,
            data.ax,
            data.ay,
            data.az,
            data.gx,
            data.gy,
            data.gz,
            data.created_at
        ],
        (err) => {
            if (err) console.error("SQL Insert Error:", err);
        }
    );

    res.json({ success: true });
});


// ------------------- LIVE SENSOR STREAM -------------------
app.get("/api/data", (req, res) => {
    res.json(sensorDataStore);
});


// ------------------- TODAY SUMMARY -------------------
app.get("/api/today-summary", (req, res) => {
    const query = `
        SELECT
            AVG(CASE WHEN bpm > 0 THEN bpm END) AS avg_bpm,

            (
                SELECT bpm_status
                FROM sensor_logs
                WHERE DATE(created_at) = CURDATE()
                AND bpm_status IN ('High', 'Normal')
                GROUP BY bpm_status
                ORDER BY COUNT(*) DESC
                LIMIT 1
            ) AS dominant_bpm_status,

            (
                SELECT repetitive
                FROM sensor_logs
                WHERE DATE(created_at) = CURDATE()
                GROUP BY repetitive
                ORDER BY COUNT(*) DESC
                LIMIT 1
            ) AS dominant_repetitive

        FROM sensor_logs
        WHERE DATE(created_at) = CURDATE();
    `;

    db.query(query, (err, rows) => {
        if (err) {
            console.error("Summary SQL Error:", err);
            return res.status(500).json({ error: "Summary Query Failed" });
        }

        const row = rows[0] || {};

        res.json({
            avg_bpm: Number(row.avg_bpm) || 0,
            dominant_bpm_status: row.dominant_bpm_status || "N/A",
            dominant_repetitive: row.dominant_repetitive || "N/A"
        });
    });
});


// ------------------- ML PREDICTION -------------------
app.post("/predict", async (req, res) => {
    try {
        const ml = await axios.post("http://127.0.0.1:5000/predict_ml", req.body);
        res.json(ml.data);
    } catch (error) {
        console.error("ML API Error:", error);
        res.status(500).json({ error: "ML Service Error" });
    }
});


// ------------------- HOME PAGE -------------------
app.get("/", (req, res) => {
    res.render("pages/index");
});


const PORT = 4000;
app.listen(PORT, () =>
    console.log(`🚀 Server running at http://localhost:${PORT}`)
);
