from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import urllib.parse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ---------------------------------------
# Load dataset
# ---------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

file_path = os.path.join(
    BASE_DIR,
    "dataset_07_student_placement_eligibility.csv"
)

df = pd.read_csv(file_path)


# ---------------------------------------
# Features
# ---------------------------------------

features = [
    "cgpa",
    "attendance_pct",
    "coding_score",
    "projects_completed",
    "internship_months",
    "backlogs"
]

X = df[features].copy()
y = df["target"]


# ---------------------------------------
# Handle missing values
# ---------------------------------------

X = X.fillna(X.median())


# ---------------------------------------
# Train-test split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# Feature scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# ---------------------------------------
# Train Logistic Regression
# ---------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------
# Web Handler
# ---------------------------------------

class handler(BaseHTTPRequestHandler):

    def do_GET(self):

        html = """
        <!DOCTYPE html>

        <html>

        <head>

            <title>
                Student Placement Eligibility
            </title>

            <style>

                body {
                    font-family: Arial, sans-serif;
                    background: #f4f6f8;
                    margin: 0;
                    padding: 40px 20px;
                }

                .container {
                    max-width: 600px;
                    margin: auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.10);
                }

                h1 {
                    text-align: center;
                    margin-bottom: 10px;
                }

                .description {
                    text-align: center;
                    color: #666;
                    margin-bottom: 25px;
                }

                label {
                    display: block;
                    margin-top: 15px;
                    font-weight: bold;
                }

                input {
                    width: 100%;
                    padding: 11px;
                    margin-top: 6px;
                    box-sizing: border-box;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    font-size: 15px;
                }

                button {
                    width: 100%;
                    padding: 12px;
                    margin-top: 25px;
                    border: none;
                    border-radius: 5px;
                    background: #222;
                    color: white;
                    font-size: 16px;
                    cursor: pointer;
                }

                button:hover {
                    background: #444;
                }

            </style>

        </head>


        <body>

            <div class="container">

                <h1>
                    Student Placement Eligibility
                </h1>

                <div class="description">
                    Enter the student's details below.
                </div>


                <form method="POST">


                    <label>
                        CGPA
                    </label>

                    <input
                        type="number"
                        name="cgpa"
                        min="4"
                        max="10"
                        step="0.01"
                        placeholder="Example: 8.20"
                        required
                    >


                    <label>
                        Attendance Percentage
                    </label>

                    <input
                        type="number"
                        name="attendance_pct"
                        min="0"
                        max="100"
                        step="0.01"
                        placeholder="Example: 85"
                        required
                    >


                    <label>
                        Coding Score
                    </label>

                    <input
                        type="number"
                        name="coding_score"
                        min="0"
                        max="10"
                        step="1"
                        placeholder="Enter coding score"
                        required
                    >


                    <label>
                        Projects Completed
                    </label>

                    <input
                        type="number"
                        name="projects_completed"
                        min="0"
                        step="1"
                        placeholder="Example: 3"
                        required
                    >


                    <label>
                        Internship Months
                    </label>

                    <input
                        type="number"
                        name="internship_months"
                        min="0"
                        step="1"
                        placeholder="Example: 6"
                        required
                    >


                    <label>
                        Backlogs
                    </label>

                    <input
                        type="number"
                        name="backlogs"
                        min="0"
                        step="1"
                        placeholder="Example: 0"
                        required
                    >


                    <button type="submit">
                        Predict Eligibility
                    </button>


                </form>

            </div>

        </body>

        </html>
        """

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html"
        )

        self.end_headers()

        self.wfile.write(
            html.encode()
        )


    def do_POST(self):

        content_length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        data = self.rfile.read(
            content_length
        ).decode()


        values = urllib.parse.parse_qs(data)


        try:

            # ---------------------------------------
            # Get user input
            # ---------------------------------------

            cgpa = float(
                values["cgpa"][0]
            )

            attendance = float(
                values["attendance_pct"][0]
            )

            coding_score = float(
                values["coding_score"][0]
            )

            projects = float(
                values["projects_completed"][0]
            )

            internship = float(
                values["internship_months"][0]
            )

            backlogs = float(
                values["backlogs"][0]
            )


            # ---------------------------------------
            # Create input DataFrame
            # ---------------------------------------

            input_data = pd.DataFrame(
                [[
                    cgpa,
                    attendance,
                    coding_score,
                    projects,
                    internship,
                    backlogs
                ]],
                columns=features
            )


            # ---------------------------------------
            # Scale input
            # ---------------------------------------

            input_scaled = scaler.transform(
                input_data
            )


            # ---------------------------------------
            # Prediction
            # ---------------------------------------

            prediction = int(
                model.predict(
                    input_scaled
                )[0]
            )


            # ---------------------------------------
            # Probabilities
            # ---------------------------------------

            probabilities = model.predict_proba(
                input_scaled
            )[0]


            not_eligible_probability = (
                probabilities[0] * 100
            )

            eligible_probability = (
                probabilities[1] * 100
            )


            # ---------------------------------------
            # Result
            # ---------------------------------------

            if prediction == 1:

                result = "Eligible for Placement"

            else:

                result = "Not Eligible for Placement"


            # ---------------------------------------
            # Result page
            # ---------------------------------------

            html = f"""
            <!DOCTYPE html>

            <html>

            <head>

                <title>
                    Prediction Result
                </title>

                <style>

                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f6f8;
                        margin: 0;
                        padding: 60px 20px;
                    }}

                    .container {{
                        max-width: 600px;
                        margin: auto;
                        background: white;
                        padding: 40px;
                        border-radius: 10px;
                        text-align: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.10);
                    }}

                    h1 {{
                        margin-bottom: 30px;
                    }}

                    .result {{
                        font-size: 26px;
                        font-weight: bold;
                        margin: 25px 0;
                    }}

                    .probability {{
                        font-size: 18px;
                        margin: 12px 0;
                    }}

                    .back {{
                        display: inline-block;
                        margin-top: 25px;
                        padding: 10px 20px;
                        background: #222;
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }}

                </style>

            </head>


            <body>

                <div class="container">

                    <h1>
                        Prediction Result
                    </h1>


                    <div class="result">
                        {result}
                    </div>


                    <div class="probability">

                        Probability of Eligible:
                        <strong>
                            {eligible_probability:.2f}%
                        </strong>

                    </div>


                    <div class="probability">

                        Probability of Not Eligible:
                        <strong>
                            {not_eligible_probability:.2f}%
                        </strong>

                    </div>


                    <a
                        class="back"
                        href="/"
                    >
                        Try Another Student
                    </a>

                </div>

            </body>

            </html>
            """


            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html"
            )

            self.end_headers()

            self.wfile.write(
                html.encode()
            )


        except Exception as error:

            self.send_response(400)

            self.send_header(
                "Content-Type",
                "text/html"
            )

            self.end_headers()


            error_page = f"""
            <html>

            <body>

                <h2>
                    Error
                </h2>

                <p>
                    {error}
                </p>

                <a href="/">
                    Go Back
                </a>

            </body>

            </html>
            """

            self.wfile.write(
                error_page.encode()
            )


# ---------------------------------------
# Run locally
# ---------------------------------------

if __name__ == "__main__":

    server = HTTPServer(
        ("localhost", 8000),
        handler
    )

    print(
        "Student Placement Eligibility application started."
    )

    print(
        "Open: http://localhost:8000"
    )

    print(
        "Press Ctrl+C to stop."
    )

    server.serve_forever()