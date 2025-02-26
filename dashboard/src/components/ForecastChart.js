import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';

const ForecastChart = () => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchPredictions();
    }, []);

    const fetchPredictions = async () => {
        try {
            const response = await axios.post('http://localhost:5000/predict', { steps: 30 });
            setData(response.data.predictions);
            setLoading(false);
        } catch (error) {
            console.error("Error fetching predictions:", error);
        }
    };

    const chartData = {
        labels: Array.from({ length: data.length }, (_, i) => `Day ${i+1}`),
        datasets: [
            {
                label: "Brent Oil Price Forecast",
                data: data,
                borderColor: "blue",
                fill: false,
            }
        ]
    };

    return (
        <div>
            <h2>Brent Oil Price Forecast (Next 30 Days)</h2>
            {loading ? <p>Loading...</p> : <Line data={chartData} />}
        </div>
    );
};

export default ForecastChart;
