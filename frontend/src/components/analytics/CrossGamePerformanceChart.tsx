import React from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

interface Metrics {
  fighting: { accuracy: number; aggression: number; defensive_ratio: number };
  badminton: { shot_variety: number; power_control: number; court_coverage: number };
  racing: { consistency: number; risk_taking: number; precision: number };
}

const CrossGamePerformanceChart = ({ metrics }: { metrics: Metrics }) => {
  const data = [
    { name: 'Fighting', Accuracy: metrics.fighting.accuracy, Aggression: metrics.fighting.aggression, Defense: metrics.fighting.defensive_ratio },
    { name: 'Badminton', Variety: metrics.badminton.shot_variety, Power: metrics.badminton.power_control, Coverage: metrics.badminton.court_coverage },
    { name: 'Racing', Consistency: metrics.racing.consistency, Risk: metrics.racing.risk_taking, Precision: metrics.racing.precision },
  ];

  return (
    <div className="w-full h-72">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff11" />
          <XAxis dataKey="name" tick={{ fill: '#cbd5e1' }} />
          <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
          <Tooltip contentStyle={{ background: '#0b1220', border: '1px solid #1f2937', color: '#e5e7eb' }} />
          <Legend />
          <Bar dataKey="Accuracy" fill="#22d3ee" />
          <Bar dataKey="Aggression" fill="#f43f5e" />
          <Bar dataKey="Defense" fill="#22c55e" />
          <Bar dataKey="Variety" fill="#a78bfa" />
          <Bar dataKey="Power" fill="#f59e0b" />
          <Bar dataKey="Coverage" fill="#10b981" />
          <Bar dataKey="Consistency" fill="#06b6d4" />
          <Bar dataKey="Risk" fill="#ef4444" />
          <Bar dataKey="Precision" fill="#84cc16" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CrossGamePerformanceChart;
