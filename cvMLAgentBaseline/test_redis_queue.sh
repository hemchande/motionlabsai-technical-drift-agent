#!/bin/bash
# Test script for Redis queue pipeline

echo "🧪 Testing Redis Queue Pipeline"
echo "=================================="
echo ""

# Check if Redis is running
echo "📊 Checking Redis connection..."
redis-cli ping > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   brew services start redis  # macOS"
    echo "   redis-server               # Linux"
    exit 1
fi
echo "✅ Redis is running"
echo ""

# Clear the queue
echo "🧹 Clearing queue..."
redis-cli DEL video_call_ended_queue > /dev/null
echo "✅ Queue cleared"
echo ""

# Start retrieval agent in background
echo "🚀 Starting Mock Retrieval Agent..."
cd "$(dirname "$0")"
python3 mock_retrieval_agent.py --timeout 1 > /tmp/retrieval_agent.log 2>&1 &
RETRIEVAL_PID=$!
echo "   PID: $RETRIEVAL_PID"
sleep 2
echo ""

# Send test messages
echo "📤 Sending test messages..."
echo ""

echo "   Test 1: Single video call"
python3 mock_video_agent.py \
  --session-id "test_session_001" \
  --activity "gymnastics" \
  --technique "back_handspring" \
  --athlete "Alice"

sleep 2

echo ""
echo "   Test 2: Multiple video calls"
python3 mock_video_agent.py --count 3

sleep 3

# Stop retrieval agent
echo ""
echo "🛑 Stopping retrieval agent..."
kill $RETRIEVAL_PID 2>/dev/null
wait $RETRIEVAL_PID 2>/dev/null

echo ""
echo "📋 Retrieval Agent Logs:"
echo "=================================="
tail -50 /tmp/retrieval_agent.log

echo ""
echo "✅ Test complete!"

