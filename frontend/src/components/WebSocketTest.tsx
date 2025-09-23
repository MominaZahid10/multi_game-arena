// frontend/src/components/WebSocketTest.tsx - UPDATED VERSION
import React from 'react';
import { useMultiGameWebSocket } from '../hooks/useMultiGameWebSocket';

export const WebSocketTest: React.FC = () => {
  const { 
    connected, 
    lastMessage, 
    connectionError, 
    sendGameAction, 
    switchGame, 
    getStatus 
  } = useMultiGameWebSocket('test-session-123');

  const testFightingAction = () => {
    const playerHealth = Math.floor(Math.random() * 50) + 50;
    const aiHealth = Math.floor(Math.random() * 50) + 50;
    
    sendGameAction('fighting', {
      action_type: 'attack',
      move_type: 'punch',
      success: Math.random() > 0.3,
      position: { x: 100, y: 200 },
      damage_dealt: Math.floor(Math.random() * 20) + 5,
      combo_count: Math.floor(Math.random() * 3) + 1,
      context: {
        player_health: playerHealth,
        ai_health: aiHealth,
        distance_to_opponent: Math.floor(Math.random() * 3) + 1,
        round: 1
      }
    });
  };

  const testBadmintonAction = () => {
    const playerScore = Math.floor(Math.random() * 15);
    const aiScore = Math.floor(Math.random() * 15);
    
    sendGameAction('badminton', {
      action_type: 'shot',
      shot_type: 'smash',
      success: Math.random() > 0.4,
      court_position: { x: 150, y: 100 },
      shuttlecock_target: { x: 350, y: 300 },
      power_level: Math.random(),
      rally_position: Math.floor(Math.random() * 10) + 1,
      context: {
        player_score: playerScore,
        ai_score: aiScore,
        rally_count: Math.floor(Math.random() * 20) + 1,
        game_point: playerScore >= 20 || aiScore >= 20
      }
    });
  };

  const testRacingAction = () => {
    const lap = Math.floor(Math.random() * 3) + 1;
    const racePosition = Math.floor(Math.random() * 5) + 1;
    
    sendGameAction('racing', {
      action_type: 'overtake',
      success: Math.random() > 0.4,
      speed: Math.floor(Math.random() * 50) + 60,
      position_on_track: { x: Math.random() * 400, y: Math.random() * 300 },
      overtaking_attempt: true,
      crash_occurred: Math.random() > 0.8,
      context: {
        lap: lap,
        race_position: racePosition,
        total_distance: Math.floor(Math.random() * 1000) + 500,
        lap_times: [Math.random() * 60 + 60, Math.random() * 60 + 60]
      }
    });
  };

  return (
    <div className="space-y-4">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-4">WebSocket Connection Test</h2>
        
        <div className={`px-4 py-2 rounded-lg mb-4 ${
          connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`}>
          Status: {connected ? 'Connected ✅' : 'Disconnected ❌'}
        </div>

        {connectionError && (
          <div className="bg-red-100 text-red-800 px-4 py-2 rounded-lg mb-4">
            Error: {connectionError}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <button 
          onClick={testFightingAction}
          disabled={!connected}
          className="px-4 py-2 bg-red-500 text-white rounded disabled:opacity-50 hover:bg-red-600"
        >
          🥊 Test Fighting
        </button>
        
        <button 
          onClick={testBadmintonAction}
          disabled={!connected}
          className="px-4 py-2 bg-green-500 text-white rounded disabled:opacity-50 hover:bg-green-600"
        >
          🏸 Test Badminton
        </button>
        
        <button 
          onClick={testRacingAction}
          disabled={!connected}
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50 hover:bg-blue-600"
        >
          🏎️ Test Racing
        </button>
        
        <button 
          onClick={getStatus}
          disabled={!connected}
          className="px-4 py-2 bg-purple-500 text-white rounded disabled:opacity-50 hover:bg-purple-600"
        >
          📊 Get Status
        </button>
      </div>

      <div className="space-y-2">
        <button 
          onClick={() => switchGame('fighting')}
          disabled={!connected}
          className="px-3 py-1 bg-gray-500 text-white rounded disabled:opacity-50 hover:bg-gray-600 mr-2"
        >
          Switch to Fighting
        </button>
        
        <button 
          onClick={() => switchGame('badminton')}
          disabled={!connected}
          className="px-3 py-1 bg-gray-500 text-white rounded disabled:opacity-50 hover:bg-gray-600 mr-2"
        >
          Switch to Badminton
        </button>
        
        <button 
          onClick={() => switchGame('racing')}
          disabled={!connected}
          className="px-3 py-1 bg-gray-500 text-white rounded disabled:opacity-50 hover:bg-gray-600"
        >
          Switch to Racing
        </button>
      </div>

      <div className="bg-gray-100 p-4 rounded-lg max-h-96 overflow-y-auto">
        <h3 className="font-bold mb-2">Last Message from Backend:</h3>
        {lastMessage ? (
          <pre className="text-xs whitespace-pre-wrap bg-white p-2 rounded border">
            {JSON.stringify(lastMessage, null, 2)}
          </pre>
        ) : (
          <div className="text-gray-500 italic">No messages yet...</div>
        )}
      </div>

      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
        <strong>Debug Info:</strong><br/>
        Session ID: test-session-123<br/>
        WebSocket URL: ws://localhost:8000/ws/multi-game/test-session-123<br/>
        Expected Backend Message Types: connection_established, analysis_update, game_switched, session_status
      </div>
    </div>
  );
};