import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { UnifiedPersonality } from '@/lib/types';

const traitMap = (
  p: UnifiedPersonality
) => ([
  { name: 'Aggression', value: p.aggression_level, color: '#ff4444' },
  { name: 'Risk Tolerance', value: p.risk_tolerance, color: '#ff8800' },
  { name: 'Analytical', value: p.analytical_thinking, color: '#4488ff' },
  { name: 'Patience', value: p.patience_level, color: '#44ff88' },
  { name: 'Precision', value: p.precision_focus, color: '#8844ff' },
  { name: 'Competitive', value: p.competitive_drive, color: '#ff44ff' },
  { name: 'Strategic', value: p.strategic_thinking, color: '#ffff44' },
  { name: 'Adaptability', value: p.adaptability, color: '#44ffff' },
]);

const UnifiedPersonalityRadar = ({ personality }: { personality: UnifiedPersonality }) => {
  const data = traitMap(personality).map(t => ({ subject: t.name, A: Math.round(t.value * 100) }));
  return (
    <div className="w-full h-80">
      <ResponsiveContainer>
        <RadarChart data={data} margin={{ top: 16, right: 16, bottom: 16, left: 16 }}>
          <PolarGrid stroke="#ffffff22" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
          <Radar name="Traits" dataKey="A" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.35} />
          <Tooltip formatter={(v: number) => `${v}%`} contentStyle={{ background: '#0b1220', border: '1px solid #1f2937', color: '#e5e7eb' }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default UnifiedPersonalityRadar;
