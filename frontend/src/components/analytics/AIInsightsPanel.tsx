import React from 'react';
import { AIInsight, AIActionResponse } from '@/lib/types';

const AIInsightsPanel = ({ lastAI }: { lastAI: AIActionResponse | null }) => {
  if (!lastAI) return (
    <div className="text-sm text-muted-foreground">No AI insights yet.</div>
  );
  return (
    <div className="space-y-2">
      <h4 className="text-lg font-semibold">AI Strategy: {lastAI.strategy}</h4>
      <p className="text-sm opacity-90">{lastAI.cross_game_reasoning}</p>
      <div className="text-xs opacity-70">Confidence: {(lastAI.confidence * 100).toFixed(1)}%</div>
    </div>
  );
};

export default AIInsightsPanel;
