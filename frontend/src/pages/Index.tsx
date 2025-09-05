import React, { useState } from 'react';
import GameLauncher from '@/components/GameLauncher';
import GameArena from '@/components/GameArena';
import AnalyticsOverlay from '@/components/AnalyticsOverlay';
import { WebSocketTest } from '@/components/WebSocketTest';

const Index = () => {
  const [currentGame, setCurrentGame] = useState<'fighting' | 'badminton' | 'racing' | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showWebSocketDebug, setShowWebSocketDebug] = useState(false); // NEW: Debug toggle

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
        
        {/* Debug button on launcher */}
        <button
          onClick={() => setShowWebSocketDebug(!showWebSocketDebug)}
          className="fixed bottom-4 right-4 px-3 py-1 bg-purple-600 text-white rounded text-xs z-50"
        >
          {showWebSocketDebug ? 'Hide' : 'Show'} WebSocket Debug
        </button>
        
        {/* WebSocket Debug Overlay */}
        {showWebSocketDebug && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-4 max-w-2xl w-full mx-4">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">WebSocket Debug</h2>
                <button
                  onClick={() => setShowWebSocketDebug(false)}
                  className="text-gray-500 hover:text-gray-700 text-2xl"
                >
                  ×
                </button>
              </div>
              <WebSocketTest />
            </div>
          </div>
        )}
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
      
      {/* NEW: WebSocket Debug Button in Game */}
      <button
        onClick={() => setShowWebSocketDebug(!showWebSocketDebug)}
        className="fixed bottom-4 right-4 px-3 py-1 bg-purple-600 text-white rounded text-xs z-40"
      >
        WS Debug
      </button>
      
      {/* NEW: WebSocket Debug Overlay in Game */}
      {showWebSocketDebug && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-4 max-w-2xl w-full mx-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">WebSocket Debug - {currentGame}</h2>
              <button
                onClick={() => setShowWebSocketDebug(false)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            <WebSocketTest />
          </div>
        </div>
      )}
    </>
  );
};

export default Index;