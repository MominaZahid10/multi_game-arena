import React from 'react';
import { WebSocketTest } from '../components/WebSocketTest';

const WebSocketTestPage = () => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">WebSocket Test Page</h1>
      <WebSocketTest />
    </div>
  );
};

export default WebSocketTestPage;