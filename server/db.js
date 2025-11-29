import mysql2 from "mysql2";

const db = mysql2.createConnection({
    host: "localhost",
    user: "root",
    password: "pranit4311e@",
    database: "autism_monitoring"
});

db.connect((err) => {
    if (err) console.error("DB Connection Failed:", err);
    else console.log("MySQL Connected ✔");
});

export default db;
