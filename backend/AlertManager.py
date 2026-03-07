"""
AlertManager — Real-Time Trade & System Notifications
======================================================
Sends alerts via email (SMTP) and/or Telegram when:
  - A live trade is executed
  - Portfolio P&L crosses a configurable threshold
  - The trading bot encounters an error
  - A stop-loss or circuit breaker is triggered

Configuration is stored in Firestore under `app_config/alerts`:
  {
      "enabled": true,
      "emailEnabled": true,
      "emailTo": "user@example.com",
      "smtpServer": "smtp.gmail.com",
      "smtpPort": 587,
      "smtpUser": "bot@example.com",
      "smtpPassword": "app-password",
      "telegramEnabled": true,
      "telegramBotToken": "123456:ABC-DEF",
      "telegramChatId": "12345678",
      "pnlThresholdPct": 3.0
  }
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AlertManager:
    """Loads config from Firestore and dispatches alerts."""

    def __init__(self, dbClient=None):
        self.db = dbClient
        self._config: Dict[str, Any] = {}
        self._loadConfig()

    def _loadConfig(self):
        if not self.db:
            return
        try:
            doc = self.db.collection('app_config').document('alerts').get()
            if doc.exists:
                self._config = doc.to_dict()
        except Exception as e:
            logger.warning(f"[AlertManager] Failed to load config: {e}")

    @property
    def enabled(self) -> bool:
        return self._config.get('enabled', False)

    def notifyTrade(self, symbol: str, side: str, qty: int, price: float,
                    status: str, confidence: float = 0.0):
        """Alert when a live trade is executed."""
        if not self.enabled:
            return
        msg = (
            f"Trade Executed: {side.upper()} {qty} {symbol}\n"
            f"Price: ${price:,.2f} | Status: {status}\n"
            f"Confidence: {confidence:.0%}"
        )
        self._send(f"Trade: {side.upper()} {symbol}", msg)

    def notifyPnLThreshold(self, currentPnLPct: float, equity: float):
        """Alert when daily P&L crosses the configured threshold."""
        if not self.enabled:
            return
        threshold = self._config.get('pnlThresholdPct', 3.0)
        if abs(currentPnLPct) >= threshold:
            direction = "gain" if currentPnLPct > 0 else "loss"
            msg = (
                f"P&L Alert: {currentPnLPct:+.2f}% daily {direction}\n"
                f"Current equity: ${equity:,.2f}\n"
                f"Threshold: {threshold}%"
            )
            self._send(f"P&L Alert: {currentPnLPct:+.2f}%", msg)

    def notifyStopLoss(self, symbol: str, lossPct: float, entryPrice: float,
                       currentPrice: float):
        """Alert when a stop-loss is triggered."""
        if not self.enabled:
            return
        msg = (
            f"STOP-LOSS triggered for {symbol}\n"
            f"Loss: {lossPct:.1f}%\n"
            f"Entry: ${entryPrice:.2f} → Current: ${currentPrice:.2f}"
        )
        self._send(f"STOP-LOSS: {symbol}", msg)

    def notifyCircuitBreaker(self, drawdownPct: float, peakEquity: float,
                             currentEquity: float):
        """Alert when portfolio circuit breaker trips."""
        if not self.enabled:
            return
        msg = (
            f"CIRCUIT BREAKER ACTIVATED\n"
            f"Drawdown: {drawdownPct:.1f}% from peak\n"
            f"Peak: ${peakEquity:,.2f} → Current: ${currentEquity:,.2f}\n"
            f"All trading halted."
        )
        self._send("CIRCUIT BREAKER", msg)

    def notifyDailyReview(self, report: dict):
        """Send end-of-day performance summary via all enabled channels."""
        if not self.enabled:
            return

        date = report.get('date', '?')
        plPct = report.get('portfolioPLPct', 0)
        pl = report.get('portfolioPL', 0)
        equity = report.get('equity', 0)
        active = report.get('activeSlots', 0)
        ghosts = report.get('ghostSlotCount', 0)

        top = report.get('topPerformers', [])
        bottom = report.get('bottomPerformers', [])
        actions = report.get('actionsTaken', [])
        ghostWatch = report.get('ghostWatch', [])
        suggestions = report.get('suggestions', [])

        lines = [
            f"Daily Review — {date}",
            f"Portfolio: {plPct:+.2f}% (${pl:+,.2f})",
            f"Equity: ${equity:,.2f}",
            f"Active: {active} slots | Ghost: {ghosts} slots",
        ]

        if top:
            best = top[0]
            lines.append(f"Top: {best['slot']} {best['recentReturn']:+.1f}%")
        if bottom:
            worst = bottom[-1]
            lines.append(f"Bottom: {worst['slot']} {worst['recentReturn']:+.1f}%")

        if actions:
            action_strs = []
            for a in actions[:3]:
                action_strs.append(f"{a['action']} {a['slot']} ({a['reason']})")
            lines.append("Actions: " + "; ".join(action_strs))

        recovering = [g for g in ghostWatch if g.get('nearRestore')]
        if recovering:
            g = recovering[0]
            lines.append(
                f"Ghost watch: {g['slot']} recovering "
                f"({g['recentReturn']:+.1f}% recent)"
            )

        if suggestions:
            lines.append(f"Suggestions: {suggestions[0]}")

        body = "\n".join(lines)
        self._send(f"Daily Review — {date}", body)

    def notifyError(self, context: str, error: str):
        """Alert on system errors."""
        if not self.enabled:
            return
        msg = f"Error in {context}:\n{error[:500]}"
        self._send(f"Error: {context}", msg)

    def _send(self, subject: str, body: str):
        """Dispatch to all enabled channels."""
        if self._config.get('emailEnabled'):
            self._sendEmail(subject, body)
        if self._config.get('telegramEnabled'):
            self._sendTelegram(f"*{subject}*\n{body}")

    def _sendEmail(self, subject: str, body: str):
        try:
            smtp_server = self._config.get('smtpServer', 'smtp.gmail.com')
            smtp_port = int(self._config.get('smtpPort', 587))
            smtp_user = self._config.get('smtpUser', '')
            smtp_pass = self._config.get('smtpPassword', '')
            email_to = self._config.get('emailTo', '')

            if not all([smtp_user, smtp_pass, email_to]):
                return

            msg = MIMEText(body)
            msg['Subject'] = f"[PortfolioBot] {subject}"
            msg['From'] = smtp_user
            msg['To'] = email_to

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        except Exception as e:
            logger.error(f"[AlertManager] Email failed: {e}")

    def _sendTelegram(self, text: str):
        try:
            import requests
            token = self._config.get('telegramBotToken', '')
            chat_id = self._config.get('telegramChatId', '')
            if not all([token, chat_id]):
                return
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown',
            }, timeout=10)
        except Exception as e:
            logger.error(f"[AlertManager] Telegram failed: {e}")
