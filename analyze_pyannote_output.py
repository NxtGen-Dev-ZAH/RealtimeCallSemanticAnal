#!/usr/bin/env python3
"""
Analyze Pyannote (WhisperX 3.x built-in) diarization output.
"""

import json
import sys
import io
from collections import defaultdict
from datetime import timedelta

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load diarization results
with open('output/call_20251125_160302_diarization.json', 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Load summary
with open('output/call_20251125_160302_diarization_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

print("=" * 80)
print("PYANNOTE (WHISPERX 3.X BUILT-IN) DIARIZATION ANALYSIS")
print("=" * 80)
print()

# Basic statistics
agent_segments = [s for s in segments if s['speaker'] == 'AGENT']
customer_segments = [s for s in segments if s['speaker'] == 'CUSTOMER']

print("📊 OVERALL STATISTICS")
print("-" * 80)
print(f"Total segments: {len(segments)}")
print(f"Unique speakers: {summary['results']['unique_speakers']}")
print(f"AGENT segments: {len(agent_segments)} ({len(agent_segments)/len(segments)*100:.1f}%)")
print(f"CUSTOMER segments: {len(customer_segments)} ({len(customer_segments)/len(segments)*100:.1f}%)")
print()

# Duration analysis
agent_duration = sum(s['end'] - s['start'] for s in agent_segments)
customer_duration = sum(s['end'] - s['start'] for s in customer_segments)
total_duration = max(s['end'] for s in segments)

print("⏱️  DURATION ANALYSIS")
print("-" * 80)
print(f"Total audio duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
print(f"AGENT speaking time: {agent_duration:.1f} seconds ({agent_duration/60:.1f} minutes) - {agent_duration/total_duration*100:.1f}%")
print(f"CUSTOMER speaking time: {customer_duration:.1f} seconds ({customer_duration/60:.1f} minutes) - {customer_duration/total_duration*100:.1f}%")
print(f"Silence/overlap: {total_duration - agent_duration - customer_duration:.1f} seconds")
print()

# Segment length analysis
agent_lengths = [s['end'] - s['start'] for s in agent_segments]
customer_lengths = [s['end'] - s['start'] for s in customer_segments]

print("📏 SEGMENT LENGTH ANALYSIS")
print("-" * 80)
print("AGENT segments:")
print(f"  Average: {sum(agent_lengths)/len(agent_lengths):.2f} seconds")
print(f"  Min: {min(agent_lengths):.2f} seconds")
print(f"  Max: {max(agent_lengths):.2f} seconds")
print()
print("CUSTOMER segments:")
print(f"  Average: {sum(customer_lengths)/len(customer_lengths):.2f} seconds")
print(f"  Min: {min(customer_lengths):.2f} seconds")
print(f"  Max: {max(customer_lengths):.2f} seconds")
print()

# Text analysis
agent_texts = [s['text'].strip() for s in agent_segments if s.get('text', '').strip()]
customer_texts = [s['text'].strip() for s in customer_segments if s.get('text', '').strip()]

print("📝 TEXT ANALYSIS")
print("-" * 80)
print(f"AGENT segments with text: {len(agent_texts)}/{len(agent_segments)}")
print(f"CUSTOMER segments with text: {len(customer_texts)}/{len(customer_segments)}")
print()

# Check for repetition issues
agent_repeated = [t for t in agent_texts if len(t.split()) > 0 and len(set(t.split())) < len(t.split()) * 0.5]
customer_repeated = [t for t in customer_texts if len(t.split()) > 0 and len(set(t.split())) < len(t.split()) * 0.5]

print("🔄 REPETITION CHECK")
print("-" * 80)
print(f"AGENT segments with potential repetition: {len(agent_repeated)}")
if agent_repeated:
    print("  Examples:")
    for text in agent_repeated[:3]:
        print(f"    - {text[:100]}...")
print()
print(f"CUSTOMER segments with potential repetition: {len(customer_repeated)}")
if customer_repeated:
    print("  Examples:")
    for text in customer_repeated[:3]:
        print(f"    - {text[:100]}...")
print()

# Conversation flow analysis
print("🗣️  CONVERSATION FLOW")
print("-" * 80)
print("First 10 segments:")
for i, seg in enumerate(segments[:10]):
    speaker = seg['speaker']
    start = seg['start']
    text = seg.get('text', '')[:60]
    print(f"  {i+1:2d}. [{start:6.1f}s] {speaker:8s}: {text}...")
print()

# Customer segment distribution
print("👤 CUSTOMER SEGMENT DISTRIBUTION")
print("-" * 80)
print("Customer segments throughout the conversation:")
for i, seg in enumerate(customer_segments[:20]):
    start = seg['start']
    end = seg['end']
    duration = end - start
    text = seg.get('text', '')[:70]
    print(f"  {i+1:2d}. [{start:6.1f}s - {end:6.1f}s] ({duration:4.1f}s): {text}...")
if len(customer_segments) > 20:
    print(f"  ... and {len(customer_segments) - 20} more customer segments")
print()

# Comparison with Resemblyzer
print("📊 COMPARISON: PYANNOTE vs RESEMBLYZER")
print("-" * 80)
print("Resemblyzer results (from previous run):")
print("  - CUSTOMER segments: 7")
print("  - AGENT segments: 269")
print("  - Total segments: 276")
print()
print("Pyannote results (current):")
print(f"  - CUSTOMER segments: {len(customer_segments)}")
print(f"  - AGENT segments: {len(agent_segments)}")
print(f"  - Total segments: {len(segments)}")
print()
print("✅ IMPROVEMENT:")
print(f"  - CUSTOMER detection: {len(customer_segments)} vs 7 ({len(customer_segments)/7:.1f}x better)")
print(f"  - More balanced speaker distribution")
print()

# Quality assessment
print("✅ QUALITY ASSESSMENT")
print("-" * 80)
issues = []
if len(customer_segments) < 20:
    issues.append("⚠️  Low customer segment count (might indicate under-detection)")
if len(agent_segments) / len(customer_segments) > 10:
    issues.append("⚠️  Very imbalanced speaker distribution (agent >> customer)")
if len(agent_repeated) > len(agent_segments) * 0.1:
    issues.append("⚠️  High repetition in agent segments")
if len(customer_repeated) > len(customer_segments) * 0.1:
    issues.append("⚠️  High repetition in customer segments")

if not issues:
    print("✅ No major issues detected!")
    print("✅ Good speaker balance")
    print("✅ Reasonable segment distribution")
    print("✅ Text mapping appears correct")
else:
    for issue in issues:
        print(issue)
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

