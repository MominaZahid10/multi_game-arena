import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import UnifiedPersonalityRadar from '@/components/analytics/UnifiedPersonalityRadar';
import CrossGamePerformanceChart from '@/components/analytics/CrossGamePerformanceChart';
import AIInsightsPanel from '@/components/analytics/AIInsightsPanel';
import { useEffect, useMemo, useState } from 'react';
import { AIActionResponse, UnifiedPersonality } from '@/lib/types';
import { connectMultiGameWS } from '@/lib/ws';

interface AnalyticsOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

const AnalyticsOverlay: React.FC<AnalyticsOverlayProps> = ({ isOpen, onClose }) => {
  const [personality, setPersonality] = useState<UnifiedPersonality | null>(null);
  const [lastAI, setLastAI] = useState<AIActionResponse | null>(null);

  const analyticsData = {
    totalGames: 147,
    winRate: 68,
    avgGameTime: '4:32',
    favoriteGame: 'Fighting',
    recentMatches: [
      { game: 'Fighting', result: 'Win', duration: '3:45', opponent: 'AI Challenger' },
      { game: 'Badminton', result: 'Loss', duration: '6:12', opponent: 'Pro Player' },
      { game: 'Racing', result: 'Win', duration: '2:33', opponent: 'Speed Demon' },
      { game: 'Fighting', result: 'Win', duration: '4:01', opponent: 'Combat Master' },
    ],
    skillLevels: { fighting: 85, badminton: 72, racing: 91 },
  };

  useEffect(() => {
    if (!isOpen) return;
    const ws = connectMultiGameWS((msg) => {
      if (msg.type === 'analysis_update') {
        setPersonality(msg.unified_personality);
        setLastAI(msg.ai_response);
      }
    });
    return () => ws.close();
  }, [isOpen]);

  const metrics = useMemo(() => ({
    fighting: { accuracy: 72, aggression: 65, defensive_ratio: 58 },
    badminton: { shot_variety: 61, power_control: 67, court_coverage: 74 },
    racing: { consistency: 70, risk_taking: 54, precision: 79 },
  }), []);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-background/95 backdrop-blur-lg z-50 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ type: 'spring', duration: 0.5 }}
            className="w-full max-w-6xl max-h-[90vh] overflow-y-auto"
          >
            <div className="flex justify-between items-center mb-6">
              <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">Arena Analytics</h1>
              <motion.button
                onClick={onClose}
                className="btn-gaming-outline px-6 py-3"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                BACK TO GAME
              </motion.button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6 mb-8">
              <Card className="bg-card/80 border border-white/10 shadow-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-foreground">Total Games</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-foreground">{analyticsData.totalGames}</div>
                </CardContent>
              </Card>
              <Card className="bg-card/80 border border-white/10 shadow-xl">
                <CardHeader className="pb-2">
                  <CardTitle className="text-foreground">Win Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-foreground">{analyticsData.winRate}%</div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
              <Card className="bg-card/80 border border-white/10 shadow-xl">
                <CardHeader>
                  <CardTitle className="text-foreground">Unified Personality</CardTitle>
                </CardHeader>
                <CardContent>
                  {personality ? (
                    <UnifiedPersonalityRadar personality={personality} />
                  ) : (
                    <div className="text-sm text-muted-foreground">Awaiting analysis updates...</div>
                  )}
                </CardContent>
              </Card>

              <Card className="bg-card/80 border border-white/10 shadow-xl">
                <CardHeader>
                  <CardTitle className="text-foreground">Cross-Game Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  <CrossGamePerformanceChart metrics={metrics} />
                </CardContent>
              </Card>
            </div>

            <Card className="bg-card/80 border border-white/10 shadow-xl mb-8">
              <CardHeader>
                <CardTitle className="text-foreground">Real-Time AI Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <AIInsightsPanel lastAI={lastAI} />
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AnalyticsOverlay;
