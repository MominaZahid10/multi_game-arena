import React, { useState, useEffect } from 'react';
import GameLauncher from '@/components/GameLauncher';
import GameArena from '@/components/GameArena';
import AnalyticsOverlay from '@/components/AnalyticsOverlay';
import { initializeSession, resetSession } from '@/lib/analytics';

const Index = () => {
  const [currentGame, setCurrentGame] = useState<'fighting' | 'badminton' | 'racing' | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(false);

  // ✅ FIX: Initialize session on app mount - force clear old sessions
  useEffect(() => {
    // Clear any old session format
    const stored = localStorage.getItem('arena_session_id');
    if (stored && stored.startsWith('session-')) {
      console.log('🔄 Clearing old session format:', stored);
      resetSession();
    }
    
    const sessionId = initializeSession();
    console.log('🚀 App initialized with session:', sessionId);
  }, []);

  const handleGameSelect = (game: 'fighting' | 'badminton' | 'racing') => {
    setCurrentGame(game);
  };

  const handleGameChange = (game: 'fighting' | 'badminton' | 'racing') => {
    setCurrentGame(game);
  };

  const handleToggleAnalytics = () => {
    setShowAnalytics(!showAnalytics);
  };

  const handleBackToLauncher = () => {
    setCurrentGame(null);
    setShowAnalytics(false);
  };

  if (!currentGame) {
    return (
      <>
        <GameLauncher onGameSelect={handleGameSelect} />
        
      </>
    );
  }

  return (
    <>
      <GameArena
        gameType={currentGame}
        onGameChange={handleGameChange}
        showAnalytics={showAnalytics}
        onToggleAnalytics={handleToggleAnalytics}
      />
      
      <AnalyticsOverlay
        isOpen={showAnalytics}
        onClose={() => setShowAnalytics(false)}
      />
      
      {/* Back to Launcher Button */}
      <button
        onClick={handleBackToLauncher}
        className="fixed top-4 right-4 btn-gaming-outline px-4 py-2 text-sm z-40"
      >
        ← MAIN MENU
      </button>
      
    </>
  );
};

export default Index;
