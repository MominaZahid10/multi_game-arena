// components/AnalyticsDashboard.tsx - COMPLETELY REWRITTEN FOR LIVE DATA

import React, { useEffect, useState } from "react";
import { fetchSessionAnalytics, fetchPersonalityProfile } from "@/lib/analytics";

type Props = {
  sessionId: string; // ✅ Required prop from parent (no fallback to avoid session mismatch)
  analytics?: any;
  personality?: any;
  onRefresh?: () => void;
  lastUpdated?: number | null;
};

const AnalyticsDashboard: React.FC<Props> = ({ 
  sessionId, // ✅ Use the session ID from the active game
  analytics: propAnalytics, 
  personality: propPersonality, 
  onRefresh, 
  lastUpdated 
}) => {
  const [liveAnalytics, setLiveAnalytics] = useState<any>(null);
  const [livePersonality, setLivePersonality] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // ✅ Fetch data on mount using the PASSED sessionId
  useEffect(() => {
    if (!sessionId) {
      console.warn('⚠️ No session ID provided to Analytics Dashboard');
      return;
    }
    
    const fetchData = async () => {
      try {
        console.log('📊 Fetching analytics for session:', sessionId);
        
        // Fetch with proper error handling
        const [aData, pData] = await Promise.all([
          fetchSessionAnalytics(sessionId).catch(err => {
            console.warn('Analytics fetch failed:', err);
            return null;
          }),
          fetchPersonalityProfile(sessionId).catch(err => {
            console.warn('Personality fetch failed:', err);
            return null;
          })
        ]);
        
        if (aData) {
          console.log('✅ Analytics received:', JSON.stringify(aData, null, 2));
          setLiveAnalytics(aData);
          setError(null);
        } else {
          console.warn('⚠️ No analytics data received');
        }
        
        if (pData) {
          console.log('✅ Personality received:', JSON.stringify(pData, null, 2));
          setLivePersonality(pData);
        } else {
          console.warn('⚠️ No personality data received');
        }
      } catch (e) {
        console.error("❌ Dashboard fetch error:", e);
        setError(e instanceof Error ? e.message : 'Unknown error');
      }
    };
    
    // Initial fetch immediately
    fetchData();
    
    // Auto-refresh every 3s
    const interval = setInterval(() => {
      console.log('🔄 Auto-refreshing analytics...');
      fetchData();
    }, 3000);
    
    return () => clearInterval(interval);
  }, [sessionId]);

  // ✅ FIX: Also update when props change
  useEffect(() => {
    if (propAnalytics) setLiveAnalytics(propAnalytics);
    if (propPersonality) setLivePersonality(propPersonality);
  }, [propAnalytics, propPersonality]);

  // Merge live data with props (live data takes priority)
  const rawAnalytics = liveAnalytics || propAnalytics || {};
  const personalityData = livePersonality || propPersonality || {};

  // ✅ ROBUST DATA EXTRACTION (Fixes "Hardcoded Data" issue)
  // The backend returns { session_info: {...}, game_breakdown: {...}, overall_stats: {...} } at root
  // OR wrapped in { session_stats: { session_info: {...}, ... } }
  const sessionInfo = 
    rawAnalytics?.session_info || 
    rawAnalytics?.session_stats?.session_info || 
    {};

  const gameBreakdown = 
    rawAnalytics?.game_breakdown || 
    rawAnalytics?.session_stats?.game_breakdown || 
    {};

  const overallStats = 
    rawAnalytics?.overall_stats || 
    rawAnalytics?.session_stats?.overall_stats || 
    {};

  // Extract personality info
  const personalityScores = personalityData?.raw_scores || personalityData || {};
  const personalityDisplay = personalityData?.personality || {};

  // Check if there's any meaningful data
  const hasData = (sessionInfo.total_actions || 0) > 0 || Object.keys(gameBreakdown).length > 0;

  // Error state
  if (error && !hasData) {
    return (
      <div className="text-center p-8">
        <div className="p-4 bg-red-900/30 border border-red-500/50 rounded-lg max-w-md mx-auto">
          <h3 className="text-lg font-semibold mb-2 text-red-400">⚠️ Error Loading Analytics</h3>
          <p className="text-sm mb-4 text-red-300">{error}</p>
          <p className="text-xs text-gray-400 mt-3">
            Make sure the backend server is running on http://localhost:8000
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="text-white p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold">📊 Live Analytics</h2>
        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-300">
            Session: <span className="font-mono text-xs">{sessionId.slice(0, 20)}...</span>
          </div>
          {lastUpdated && (
            <div className="text-sm text-gray-400">
              Updated: {new Date(lastUpdated).toLocaleTimeString()}
            </div>
          )}
          <button
            onClick={() => {
              onRefresh?.();
            }}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm font-medium transition-colors"
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* No Data Message */}
      {!hasData && (
        <div className="p-8 bg-gray-800/50 border border-gray-700 rounded-lg text-center mb-6">
          <div className="text-6xl mb-4">🎮</div>
          <h3 className="text-xl font-bold mb-2">No Activity Yet</h3>
          <p className="text-gray-400">
            Start playing a game to see your analytics and personality insights!
          </p>
        </div>
      )}

      {/* Session Stats */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-xl font-bold mb-3">🎮 Session Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 bg-gray-700 rounded">
            <div className="text-sm text-gray-400">Total Actions</div>
            <div className="text-2xl font-bold text-green-400">
              {sessionInfo.total_actions || overallStats.total_actions || 0}
            </div>
          </div>
          <div className="p-3 bg-gray-700 rounded">
            <div className="text-sm text-gray-400">Games Played</div>
            <div className="text-2xl font-bold text-blue-400">
              {sessionInfo.games_played?.length || overallStats.games_played_count || 0}
            </div>
          </div>
          <div className="p-3 bg-gray-700 rounded">
            <div className="text-sm text-gray-400">Success Rate</div>
            <div className="text-2xl font-bold text-purple-400">
              {((overallStats.success_rate || 0) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="p-3 bg-gray-700 rounded">
            <div className="text-sm text-gray-400">Current Game</div>
            <div className="text-lg font-bold text-yellow-400 capitalize">
              {sessionInfo.current_game || 'None'}
            </div>
          </div>
        </div>
      </div>

      {/* Game Breakdown */}
      {Object.keys(gameBreakdown).length > 0 && (
        <div className="mb-6 p-4 bg-gray-800 rounded-lg">
          <h3 className="text-xl font-bold mb-3">🎯 Per-Game Stats</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(gameBreakdown).map(([game, stats]: [string, any]) => (
              <div key={game} className="p-4 bg-gray-700 rounded">
                <div className="text-lg font-bold mb-2 capitalize">{game}</div>
                <div className="space-y-1 text-sm">
                  <div>Actions: <span className="font-bold">{stats.total_actions}</span></div>
                  <div>Success: <span className="font-bold text-green-400">
                    {(stats.success_rate * 100).toFixed(0)}%
                  </span></div>
                  {stats.last_played && (
                    <div className="text-xs text-gray-400">
                      Last: {new Date(stats.last_played).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Personality Profile */}
      <div className="p-4 bg-gray-800 rounded-lg">
        <h3 className="text-xl font-bold mb-3">🧠 Personality Analysis</h3>
        
        {/* Archetype Display - from fixed backend */}
        {(personalityData?.archetype || personalityData?.personality_type) && (
          <div className="mb-4 p-4 bg-gradient-to-r from-purple-900/50 to-cyan-900/50 rounded-lg border border-purple-500/30">
            <div className="text-xl font-bold text-cyan-400 mb-2">
              {personalityData.archetype || personalityData.personality_type || 'Analyzing...'}
            </div>
            {personalityData.description && (
              <div className="text-sm text-gray-300">
                {personalityData.description}
              </div>
            )}
            {personalityData.status === 'no_data' && (
              <div className="text-xs text-yellow-400 mt-2">
                ⏳ Keep playing to reveal your personality!
              </div>
            )}
          </div>
        )}

        {/* Legacy format support */}
        {personalityDisplay.impressive_categories && !personalityData?.archetype && (
          <div className="mb-4 p-3 bg-gray-700 rounded">
            <div className="text-lg font-bold text-cyan-400 mb-1">
              {personalityDisplay.impressive_categories.personality_type}
            </div>
            <div className="text-sm text-gray-300">
              {personalityDisplay.impressive_categories.playstyle}
            </div>
            <div className="text-xs text-gray-400 mt-2">
              Confidence: {(personalityDisplay.impressive_categories.category_confidence * 100).toFixed(0)}%
            </div>
          </div>
        )}

        {/* Personality Traits - prioritize new format */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(personalityData?.personality || personalityData?.raw_scores || personalityScores || {}).map(([trait, value]: [string, any]) => {
            if (typeof value !== 'number') return null;
            
            const percentage = (value * 100).toFixed(0);
            const barColor = value > 0.7 ? 'bg-green-500' : value > 0.4 ? 'bg-cyan-500' : 'bg-yellow-500';
            
            // Pretty names for traits
            const traitNames: Record<string, string> = {
              aggression: '⚔️ Aggression',
              aggression_level: '⚔️ Aggression',
              patience: '🧘 Patience',
              patience_level: '🧘 Patience',
              strategic_thinking: '🧠 Strategy',
              risk_tolerance: '🎲 Risk Taking',
              precision_focus: '🎯 Precision',
              adaptability: '🔄 Adaptability',
              competitive_drive: '🏆 Competitive',
              analytical_thinking: '📊 Analytical'
            };
            
            return (
              <div key={trait} className="p-3 bg-gray-700/50 rounded border border-gray-600">
                <div className="text-xs text-gray-400 mb-1">
                  {traitNames[trait] || trait.replace(/_/g, ' ')}
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-3 bg-gray-600 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${barColor} transition-all duration-500`}
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <div className="text-sm font-bold w-12 text-right">{percentage}%</div>
                </div>
              </div>
            );
          })}
        </div>
        
        {/* No personality data message */}
        {!personalityData?.personality && !personalityData?.raw_scores && Object.keys(personalityScores).length === 0 && (
          <div className="text-center p-4 text-gray-400">
            <div className="text-4xl mb-2">🎮</div>
            <p>Play more to analyze your personality!</p>
            <p className="text-xs mt-1">ML analysis runs every 10 actions</p>
          </div>
        )}
      </div>

      {/* Raw Data (Debug) */}
      <details className="mt-4 text-sm text-gray-400">
        <summary className="cursor-pointer hover:text-white">🔍 Debug: Raw Data</summary>
        <pre className="mt-2 p-3 bg-black rounded overflow-auto max-h-64 text-xs">
          {JSON.stringify({ rawAnalytics, personalityData }, null, 2)}
        </pre>
      </details>
    </div>
  );
};

export default AnalyticsDashboard;