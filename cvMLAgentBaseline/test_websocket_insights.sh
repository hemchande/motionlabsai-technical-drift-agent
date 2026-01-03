#!/bin/bash
# Test script for WebSocket insights server

echo "🧪 Testing WebSocket Insights Server"
echo "======================================"

# Start server in background
echo "🚀 Starting WebSocket server..."
cd "$(dirname "$0")"
python3 websocket_insights_server.py --host localhost --port 8765 &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Test 1: Monitor follow-up
echo ""
echo "📤 Test 1: Sending Monitor follow-up"
python3 websocket_insights_client.py \
  --uri "ws://localhost:8765" \
  --insight "Insufficient height off floor/beam" \
  --follow-up "Monitor"

sleep 1

# Test 2: Adjust Training follow-up
echo ""
echo "📤 Test 2: Sending Adjust Training follow-up"
python3 websocket_insights_client.py \
  --uri "ws://localhost:8765" \
  --insight "Insufficient landing knee extension" \
  --follow-up "Adjust Training"

sleep 1

# Test 3: Escalate follow-up
echo ""
echo "📤 Test 3: Sending Escalate to AT/PT follow-up"
python3 websocket_insights_client.py \
  --uri "ws://localhost:8765" \
  --insight "Knee valgus collapse (inward collapse)" \
  --follow-up "Escalate to AT/PT"

sleep 1

# Test 4: Dismiss follow-up
echo ""
echo "📤 Test 4: Sending Dismiss follow-up"
python3 websocket_insights_client.py \
  --uri "ws://localhost:8765" \
  --insight "Poor landing quality" \
  --follow-up "Dismiss"

# Kill server
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null

echo ""
echo "✅ Tests complete!"

