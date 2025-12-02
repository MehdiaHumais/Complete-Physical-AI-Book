import React, { useState } from 'react';

export default function AiTutor() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello! I am Dr. Robot. Ask me anything about the book.' }
  ]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!query.trim()) return;
    
    // Add user message to UI
    const userMsg = { role: 'user', text: query };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    setQuery('');

    try {
      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMsg.text })
      });
      
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'bot', text: data.reply }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: 'Error: Is the python server running?' }]);
    }
    setLoading(false);
  };

  return (
    <div style={{ position: 'fixed', bottom: 20, right: 20, zIndex: 9999, fontFamily: 'sans-serif' }}>
      {/* 1. Toggle Button */}
      {!isOpen && (
        <button 
          onClick={() => setIsOpen(true)}
          style={{ width: '60px', height: '60px', borderRadius: '50%', background: '#25c2a0', border: 'none', cursor: 'pointer', boxShadow: '0 4px 12px rgba(0,0,0,0.3)', fontSize: '24px' }}
        >
          🤖
        </button>
      )}

      {/* 2. Chat Window */}
      {isOpen && (
        <div style={{ width: '350px', height: '500px', background: 'white', borderRadius: '12px', display: 'flex', flexDirection: 'column', boxShadow: '0 4px 20px rgba(0,0,0,0.2)', border: '1px solid #ddd' }}>
          
          {/* Header */}
          <div style={{ padding: '15px', background: '#25c2a0', color: 'white', borderTopLeftRadius: '12px', borderTopRightRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong>Physical AI Tutor</strong>
            <button onClick={() => setIsOpen(false)} style={{ background: 'none', border: 'none', color: 'white', fontSize: '16px', cursor: 'pointer' }}>✖</button>
          </div>

          {/* Messages Area */}
          <div style={{ flex: 1, padding: '15px', overflowY: 'auto', background: '#f9f9f9', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {messages.map((m, i) => (
              <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', background: m.role === 'user' ? '#25c2a0' : 'white', color: m.role === 'user' ? 'white' : 'black', padding: '10px', borderRadius: '8px', maxWidth: '80%', border: m.role === 'bot' ? '1px solid #ddd' : 'none', fontSize: '14px' }}>
                {m.text}
              </div>
            ))}
            {loading && <div style={{ alignSelf: 'flex-start', color: '#888', fontSize: '12px' }}>Thinking...</div>}
          </div>

          {/* Input Area */}
          <div style={{ padding: '10px', borderTop: '1px solid #eee', display: 'flex', gap: '10px' }}>
            <input 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask a question..."
              style={{ flex: 1, padding: '10px', borderRadius: '6px', border: '1px solid #ccc', outline: 'none' }}
            />
            <button onClick={sendMessage} style={{ padding: '10px 15px', background: '#25c2a0', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
}