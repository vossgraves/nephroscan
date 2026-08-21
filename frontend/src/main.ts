import { analyzeImage, chat, getAiHealth } from './api.js';
import type { ApiError, ChatContext, ImageAnalysisResponse, ReportContext } from './contracts.js';
import { getChatElements, getImageInput, getReportBody, escapeHtml, readReportContext, setStatus } from './dom.js';
import type { ChatElements } from './dom.js';
import { addChatMessage, createFrontendState, imageRequestKey, resetChat, type AiCapability } from './state.js';

declare global {
  interface Window {
    __nephroTypedChat?: boolean;
    __nephroLocalChatReply?: (text: string) => string | null;
  }
}

const state = createFrontendState();
const EDUCATIONAL_DISCLAIMER = 'Optional AI assistance only. This educational screening prototype is not a diagnosis and does not replace review by a qualified healthcare professional.';
let reportMutationTimer: number | undefined;
let aiInitialization: Promise<void> | undefined;

async function initializeAi(): Promise<void> {
  try {
    const health = await getAiHealth();
    state.aiCapability = health.enabled ? 'enabled' : 'disabled';
  } catch {
    state.aiCapability = 'unavailable';
    aiInitialization = undefined;
  }
  const reportBody = getReportBody();
  const reportText = reportBody?.textContent?.trim() || '';
  if (!reportBody || reportBody.querySelector('.placeholder') || !reportText || reportText.includes('Upload scans') || reportText.includes('Analysis in progress') || reportText.includes('Analysis Error')) return;
  void enrichCurrentReport();
}
function ensureAiInitialized(): Promise<void> {
  if (!aiInitialization) aiInitialization = initializeAi();
  return aiInitialization;
}


function isApiError(error: unknown): error is ApiError {
  return error instanceof Error && error.name === 'ApiError';
}

interface LocalReviewContext {
  prediction: string;
  confidence: number;
  threshold: number;
}

function readLocalReviewContext(reportBody: HTMLElement): LocalReviewContext | null {
  const confidence = Number(reportBody.dataset.localConfidence);
  const threshold = Number(reportBody.dataset.localReviewThreshold || 60);
  const prediction = reportBody.dataset.localPrediction || '';
  if (!prediction || !Number.isFinite(confidence) || !Number.isFinite(threshold)) return null;
  return { prediction, confidence, threshold };
}

function renderAiCapabilityNotice(capability: AiCapability): void {
  const panel = document.getElementById('aiOutagePanel');
  if (!panel) return;
  const down = capability === 'disabled' || capability === 'unavailable';
  panel.hidden = !down;
  panel.setAttribute('aria-hidden', String(!down));
  panel.dataset.aiCapability = capability;
}

function markAiUnavailable(error: unknown): void {
  if (isApiError(error) && error.disabled) {
    state.aiCapability = 'disabled';
    renderAiCapabilityNotice('disabled');
    return;
  }
  if (isApiError(error) && (error.status === 429 || error.status === 504)) return;
  if (isApiError(error) && error.status >= 400 && error.status < 500) return;
  state.aiCapability = 'unavailable';
  // Drop the health probe memo after a real outage so the next request can re-probe.
  aiInitialization = undefined;
  renderAiCapabilityNotice('unavailable');
}

function imageStatusElement(reportBody: HTMLElement): HTMLElement {
  const existing = reportBody.querySelector<HTMLElement>('[data-ai-image-status]');
  if (existing) return existing;
  const status = document.createElement('div');
  status.className = 'disclaimer-box';
  status.dataset.aiImageStatus = 'true';
  status.dataset.aiStatus = 'idle';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  reportBody.appendChild(status);
  return status;
}

function renderHybridCallout(reportBody: HTMLElement, stateName: 'pending' | 'ready' | 'fallback', message: string, prediction?: string, confidence?: number, threshold?: number): void {
  reportBody.querySelectorAll('[data-hybrid-callout]').forEach((callout) => callout.remove());
  const callout = document.createElement('div');
  callout.className = 'hybrid-callout';
  callout.dataset.hybridCallout = 'true';
  callout.dataset.state = stateName;
  callout.setAttribute('role', 'status');
  const head = document.createElement('div');
  head.className = 'hybrid-callout-head';
  const icon = document.createElement('i');
  icon.className = stateName === 'fallback' ? 'fas fa-triangle-exclamation' : stateName === 'pending' ? 'fas fa-circle-notch fa-spin' : 'fas fa-route';
  icon.setAttribute('aria-hidden', 'true');
  const title = document.createElement('strong');
  title.textContent = stateName === 'fallback' ? 'HYBRID REVIEW UNAVAILABLE' : stateName === 'pending' ? 'HYBRID REVIEW' : 'HYBRID REVIEW ATTACHED';
  const badge = document.createElement('span');
  badge.className = 'hybrid-callout-badge';
  badge.textContent = stateName === 'fallback' ? 'LOCAL RETAINED' : stateName === 'pending' ? 'SUPPLEMENTAL' : 'LOCAL PRIMARY';
  head.append(icon, title, badge);
  const body = document.createElement('div');
  body.className = 'hybrid-callout-body';
  body.textContent = prediction && confidence !== undefined && threshold !== undefined
    ? `${message} Local prediction: ${prediction} (${confidence}% confidence; ${threshold}% review threshold).`
    : message;
  callout.append(head, body);
  const status = imageStatusElement(reportBody);
  reportBody.insertBefore(callout, status);
}

function renderImageAnalysis(reportBody: HTMLElement, result: ImageAnalysisResponse, review?: LocalReviewContext): void {
  reportBody.querySelector('[data-ai-image-enhancement]')?.remove();
  reportBody.querySelectorAll('[data-hybrid-callout]').forEach((callout) => callout.remove());
  const section = document.createElement('section');
  section.className = 'report-section ai-image-enhancement';
  section.dataset.aiImageEnhancement = 'true';
  const findings = result.findings.filter((item) => item.trim().length > 0);
  const limitations = result.limitations.filter((item) => item.trim().length > 0);
  const nextSteps = result.next_steps.filter((item) => item.trim().length > 0);
  const list = (items: string[]): string => items.length
    ? `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
    : '<p>No additional observations were returned.</p>';
  const hybridLead = review
    ? `<div class="hybrid-callout" data-hybrid-callout="true" data-state="ready" role="status"><div class="hybrid-callout-head"><i class="fas fa-route" aria-hidden="true"></i><strong>HYBRID REVIEW</strong><span class="hybrid-callout-badge">LOCAL PRIMARY</span></div><div class="hybrid-callout-body">Local model prediction <b>${escapeHtml(review.prediction)}</b> was retained, but its ${review.confidence}% confidence is below the ${review.threshold}% review threshold. The API review below is supplemental and is not an accuracy or diagnosis claim.</div></div>`
    : '';
  section.innerHTML = `
    <div class="section-title"><i class="fas fa-sparkles"></i> OPTIONAL MULTIMODAL REVIEW</div>
    ${hybridLead}
    <div class="content">
      <p><strong>Summary:</strong> ${escapeHtml(result.summary || 'No summary returned.')}</p>
      <p><strong>Candidate observations:</strong></p>${list(findings)}
      ${limitations.length ? `<p><strong>Limitations:</strong></p>${list(limitations)}` : ''}
      ${nextSteps.length ? `<p><strong>Suggested discussion points:</strong></p>${list(nextSteps)}` : ''}
      <div class="disclaimer-box">${escapeHtml(result.disclaimer || EDUCATIONAL_DISCLAIMER)}</div>
      <div class="ai-meta">${review ? `<span><i class="fas fa-route"></i> Local primary: ${escapeHtml(review.prediction)}</span><span><i class="fas fa-gauge-high"></i> ${review.confidence}% confidence · ${review.threshold}% review threshold</span>` : ''}<span><i class="fas fa-cloud"></i> ${escapeHtml(result.provider)}</span><span><i class="fas fa-microchip"></i> ${escapeHtml(result.model)}</span><span><i class="fas fa-hashtag"></i> ${escapeHtml(result.request_id)}</span></div>
    </div>`;
  const status = imageStatusElement(reportBody);
  setStatus(status, 'ready', review ? 'Hybrid API review added; the local model prediction remains primary.' : 'Optional multimodal review added. The existing local report remains the primary screening output.');
  renderAiCapabilityNotice('enabled');
  markHybridReport(reportBody, review);
  reportBody.insertBefore(section, status);
}

function markHybridReport(reportBody: HTMLElement, review?: LocalReviewContext): void {
  if (!review) return;
  const notice = reportBody.querySelector<HTMLElement>('#uncertainResultNotice');
  if (!notice) return;
  notice.dataset.hybridReviewed = 'true';
  const strong = document.createElement('strong');
  const icon = document.createElement('i');
  icon.className = 'fas fa-route';
  icon.setAttribute('aria-hidden', 'true');
  strong.append(icon, document.createTextNode(' HYBRID REVIEW ATTACHED'));
  const prediction = document.createElement('b');
  prediction.textContent = review.prediction;
  notice.replaceChildren(
    strong,
    document.createElement('br'),
    document.createTextNode('The local model result '),
    prediction,
    document.createTextNode(` remains the primary prediction. Its ${review.confidence}% confidence was below the ${review.threshold}% review threshold, so supplemental API analysis was requested. This does not establish accuracy or diagnosis.`),
  );
}

function renderImageUnavailable(reportBody: HTMLElement, message: string, status: 'disabled' | 'error', disclaimer = EDUCATIONAL_DISCLAIMER, includeLocalReport = true): void {
  reportBody.querySelector('[data-ai-image-enhancement]')?.remove();
  const element = imageStatusElement(reportBody);
  const suffix = includeLocalReport ? ' Existing local analysis is still available.' : '';
  const copy = `${message}${suffix} ${disclaimer}`;
  if (element.textContent !== copy) setStatus(element, status, copy);
}

async function fileFromPreview(): Promise<File | null> {
  const image = document.querySelector<HTMLImageElement>('#scanPreviewWrap img');
  const source = image?.getAttribute('src');
  if (!source || !source.startsWith('blob:')) return null;
  try {
    const response = await fetch(source);
    const blob = await response.blob();
    if (!blob.size || !blob.type.startsWith('image/')) return null;
    return new File([blob], 'scan-preview', { type: blob.type });
  } catch {
    return null;
  }
}

async function enrichCurrentReport(): Promise<void> {
  const reportBody = getReportBody();
  if (!reportBody || reportBody.querySelector('[data-ai-image-enhancement]')) return;
  const reportText = (reportBody.textContent || '').trim();
  if (isExploratoryScan(readReportContext().scanType)) return;
  if (!reportText || reportBody.querySelector('.placeholder') || reportText.includes('Upload scans') || reportText.includes('Analysis in progress') || reportText.includes('Analysis Error')) return;
  const review = readLocalReviewContext(reportBody);
  if (!review || review.confidence >= review.threshold) return;
  if (state.aiCapability === 'unknown') return;
  if (state.aiCapability !== 'enabled') {
    const unavailableMessage = state.aiCapability === 'disabled' ? 'Optional multimodal review is disabled on this server.' : 'Optional multimodal review is unavailable right now.';
    renderImageUnavailable(reportBody, unavailableMessage, state.aiCapability === 'disabled' ? 'disabled' : 'error');
    if (!reportBody.querySelector('[data-hybrid-callout]')) {
      renderHybridCallout(reportBody, 'fallback', `${unavailableMessage} The local prediction remains available; no supplemental review was added.`, review.prediction, review.confidence, review.threshold);
    }
    return;
  }
  const image = state.selectedImage || await fileFromPreview();
  if (!image) return;
  state.selectedImage = image;
  const baseContext = readReportContext();
  const context: ReportContext = {
    scanType: baseContext.scanType,
    clinicalIndication: '',
    symptoms: '',
    patientName: '',
    patientId: '',
    reportText: '',
    localPrediction: review.prediction,
    localConfidence: review.confidence,
    localReviewThreshold: review.threshold,
  };
  const key = `${imageRequestKey(image, context)}:${review.prediction}:${review.confidence}:${review.threshold}`;
  if (state.imageRequestKey === key || state.imageStatus === 'loading' || state.imageStatus === 'error') return;
  state.imageRequestKey = key;
  state.imageStatus = 'loading';
  const status = imageStatusElement(reportBody);
  renderHybridCallout(reportBody, 'pending', `Reviewing this low-confidence result (${review.confidence}% < ${review.threshold}%). The local prediction remains primary.`, review.prediction, review.confidence, review.threshold);
  setStatus(status, 'loading', `AI is analyzing this low-confidence result (${review.confidence}% < ${review.threshold}%). The local prediction remains available.`);
  try {
    const result = await analyzeImage(image, context.scanType, context);
    state.imageStatus = 'ready';
    renderImageAnalysis(reportBody, result, review);
  } catch (error) {
    markAiUnavailable(error);
    const imageStatus: 'disabled' | 'error' = isApiError(error) && error.disabled ? 'disabled' : 'error';
    state.imageStatus = imageStatus;
    state.imageRequestKey = '';
    const message = isApiError(error) && error.disabled
      ? 'Optional multimodal review is disabled on this server.'
      : isApiError(error) && error.status === 429
        ? 'The AI provider is rate limiting multimodal requests; please retry shortly.'
        : 'Optional multimodal review is unavailable right now.';
    const disclaimer = isApiError(error) && error.disclaimer ? error.disclaimer : EDUCATIONAL_DISCLAIMER;
    renderImageUnavailable(reportBody, message, imageStatus, disclaimer);
    renderHybridCallout(reportBody, 'fallback', `${message} The local prediction remains available; no supplemental review was added.`);
}

}
function observeReports(): void {
  const reportBody = getReportBody();
  if (!reportBody) return;
  const observer = new MutationObserver(() => {
    window.clearTimeout(reportMutationTimer);
    reportMutationTimer = window.setTimeout(() => void enrichCurrentReport(), 120);
  });
  observer.observe(reportBody, { childList: true, subtree: true, characterData: true });
}

function isExploratoryScan(scanType: string): boolean {
  return scanType === 'other' || scanType === 'other_ai';
}

async function runExploratoryAnalysis(button: HTMLButtonElement): Promise<void> {
  const reportBody = getReportBody();
  const scanType = (document.getElementById('scanType') as HTMLSelectElement | null)?.value || '';
  if (!reportBody || !isExploratoryScan(scanType)) return;
  await ensureAiInitialized();
  if (state.aiCapability !== 'enabled') {
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, state.aiCapability === 'disabled' ? 'Optional multimodal review is disabled on this server.' : 'Optional multimodal review is unavailable right now.', state.aiCapability === 'disabled' ? 'disabled' : 'error', EDUCATIONAL_DISCLAIMER, false);
    return;
  }
  const image = state.selectedImage || await fileFromPreview();
  if (!image) {
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, 'Upload a supported image before running exploratory analysis.', 'error', EDUCATIONAL_DISCLAIMER, false);
    return;
  }
  state.selectedImage = image;
  state.imageStatus = 'loading';
  state.imageRequestKey = `exploratory:${Date.now()}`;
  reportBody.innerHTML = '<div class="loading-report"><div class="spinner-ring"></div><p><i class="fas fa-microchip"></i> Optional multimodal review in progress</p><div style="font-size:.68rem;color:#9aa7b2;">No local classifier is used for this scan type.</div></div>';
  const status = imageStatusElement(reportBody);
  setStatus(status, 'loading', 'Optional multimodal review is loading. This is not a diagnosis.');
  const originalLabel = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Reviewing...';
  try {
    const baseContext = readReportContext();
    const imageContext: ReportContext = {
      scanType: baseContext.scanType,
      clinicalIndication: '',
      symptoms: '',
      patientName: '',
      patientId: '',
      reportText: '',
    };
    const result = await analyzeImage(image, scanType, imageContext);
    state.imageStatus = 'ready';
    reportBody.innerHTML = '';
    renderImageAnalysis(reportBody, result);
  } catch (error) {
    markAiUnavailable(error);
    const imageStatus: 'disabled' | 'error' = isApiError(error) && error.disabled ? 'disabled' : 'error';
    state.imageStatus = imageStatus;
    const message = isApiError(error) && error.disabled
      ? 'Optional multimodal review is disabled on this server.'
      : 'Optional multimodal review is unavailable right now.';
    const disclaimer = isApiError(error) && error.disclaimer ? error.disclaimer : EDUCATIONAL_DISCLAIMER;
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, message, imageStatus, disclaimer, false);
  } finally {
    button.disabled = false;
    button.innerHTML = originalLabel;
  }
}

function connectExploratoryAnalysis(): void {
  const button = document.getElementById('generateBtn') as HTMLButtonElement | null;
  const scanType = document.getElementById('scanType') as HTMLSelectElement | null;
  if (!button || !scanType) return;
  button.addEventListener('click', (event) => {
    if (!isExploratoryScan(scanType.value)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void runExploratoryAnalysis(button);
  }, true);
}

function captureSelectedImage(): void {
  const input = getImageInput();
  if (!input) return;
  input.addEventListener('change', () => {
    const file = Array.from(input.files || []).find((candidate) => candidate.type.startsWith('image/'));
    if (file) {
      state.selectedImage = file;
      state.imageRequestKey = '';
      state.imageStatus = 'idle';
    }
  }, true);
  document.getElementById('clearBtn')?.addEventListener('click', () => {
    state.selectedImage = null;
    state.imageRequestKey = '';
    state.imageStatus = 'idle';
  }, true);
}

function chatContext(): ChatContext {
  const context = readReportContext();
  const reportBody = getReportBody();
  return {
    scan_type: context.scanType || 'unspecified',
    local_prediction: reportBody?.dataset.localPrediction || 'not available',
    local_confidence: reportBody?.dataset.localConfidence || 'not available',
    local_review_threshold: reportBody?.dataset.localReviewThreshold || 'not available',
    report_available: String(Boolean(context.reportText)),
  };
}

function chatStatusElement(elements: ChatElements): HTMLElement {
  const parent = elements.messages.parentElement || elements.messages;
  const existing = parent.querySelector<HTMLElement>('[data-ai-chat-status]');
  if (existing) return existing;
  const status = document.createElement('div');
  status.dataset.aiChatStatus = 'true';
  status.style.cssText = 'font-size:.68rem;color:#6b7f8d;margin:.4rem 0;';
  parent.insertBefore(status, elements.messages.nextSibling);
  return status;
}

function collapseGuidedPrompts(elements: ChatElements, collapsed: boolean): void {
  const grid = elements.quick.querySelector<HTMLElement>('.hg-prompt-grid');
  if (!grid) return;
  if (collapsed && grid.contains(document.activeElement)) elements.input.focus();
  grid.classList.toggle('is-collapsed', collapsed);
  elements.quick.querySelector<HTMLButtonElement>('#healthGuidePromptsToggle')?.setAttribute('aria-expanded', String(!collapsed));
  grid.setAttribute('aria-hidden', String(collapsed));
  grid.querySelectorAll<HTMLButtonElement>('button').forEach((button) => { button.tabIndex = collapsed ? -1 : 0; });
}

function setChatLoading(elements: ChatElements, loading: boolean, label = 'AI is analyzing your question…'): void {
  elements.messages.querySelector('[data-ai-chat-loading]')?.remove();
  if (!loading) return;
  const message = document.createElement('div');
  message.className = 'bot-msg bot hg-loading';
  message.dataset.aiChatLoading = 'true';
  message.setAttribute('role', 'status');
  message.setAttribute('aria-live', 'polite');
  for (let index = 0; index < 3; index += 1) {
    const dot = document.createElement('span');
    dot.className = 'loading-dot';
    dot.setAttribute('aria-hidden', 'true');
    message.appendChild(dot);
  }
  const copy = document.createElement('span');
  copy.className = 'hg-loading-label';
  copy.textContent = label;
  message.appendChild(copy);
  elements.messages.appendChild(message);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function renderAnimatedChatText(element: HTMLElement, text: string): void {
  element.replaceChildren();
  let wordIndex = 0;
  text.split(/(\s+)/).forEach((part) => {
    if (!part) return;
    if (/^\s+$/.test(part)) element.appendChild(document.createTextNode(part));
    else {
      const word = document.createElement('span');
      word.className = 'hg-word';
      word.style.setProperty('--word-index', String(Math.min(wordIndex, 35)));
      word.textContent = part;
      element.appendChild(word);
      wordIndex += 1;
    }
  });
}

function addChatAttribution(element: HTMLElement): void {
  const footer = document.createElement('small');
  footer.className = 'chat-attribution';
  footer.textContent = 'AI-generated guidance · verify important decisions with a qualified professional.';
  element.appendChild(footer);
}

function renderApiChatText(element: HTMLElement, text: string): void {
  const lines = text.split(/\r?\n/);
  type ApiChatBlock = { kind: 'preamble' | 'answer' | 'consider' | 'next'; label?: string; lines: string[] };
  const blocks: ApiChatBlock[] = [];
  const preamble: string[] = [];
  let current: ApiChatBlock | null = null;
  // Providers sometimes wrap the required headings in Markdown list, quote,
  // heading, or emphasis markers. Strip those only from the heading line so
  // the literal Answer/Consider/Next step contract remains recognizable.
  const heading = /^\s*(?:(?:#{1,6}|>+|[-*+•])\s*|\d+[.)]\s*)*(?:[*_~`]{1,3}\s*)?(answer|consider|next(?:[-\s]+step)?)\s*(?:[*_~`]{1,3})?\s*:\s*(.*)$/i;
  for (const line of lines) {
    const match = heading.exec(line);
    if (match) {
      const rawKind = match[1].toLowerCase();
      const kind = rawKind.startsWith('next') ? 'next' : rawKind as 'answer' | 'consider';
      current = { kind, label: kind === 'next' ? 'Next step' : kind[0].toUpperCase() + kind.slice(1), lines: match[2] ? [match[2]] : [] };
      blocks.push(current);
    } else if (current) {
      current.lines.push(line);
    } else {
      preamble.push(line);
    }
  }
  if (blocks.length && preamble.some((line) => line.trim())) {
    blocks.unshift({ kind: 'preamble', lines: preamble });
  }
  element.replaceChildren();
  if (!blocks.length) {
    renderAnimatedChatText(element, text);
    return;
  }
  for (const block of blocks) {
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-block';
    wrapper.dataset.kind = block.kind;
    const content = document.createElement('span');
    renderAnimatedChatText(content, block.lines.join('\n').trim());
    if (block.label) {
      const label = document.createElement('strong');
      label.textContent = block.label;
      wrapper.append(label, content);
    } else {
      wrapper.append(content);
    }
    element.appendChild(wrapper);
  }
}

function appendChatBubble(elements: ChatElements, role: 'user' | 'bot', text: string): void {
  const bubble = document.createElement('div');
  bubble.className = `bot-msg ${role}`;
  if (role === 'bot') {
    bubble.dataset.source = 'local';
    renderAnimatedChatText(bubble, text);
  } else {
    bubble.textContent = text;
  }
  elements.messages.appendChild(bubble);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function replaceLatestLocalReply(elements: ChatElements, text: string, source: 'local' | 'api' = 'local'): void {
  const reply = document.createElement('div');
  reply.className = 'bot-msg bot';
  reply.dataset.source = source;
  if (source === 'api') {
    renderApiChatText(reply, text);
    addChatAttribution(reply);
  } else {
    renderAnimatedChatText(reply, text);
  }
  elements.messages.appendChild(reply);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}
type LocalFallbackResult = 'reply' | 'cleared' | 'none';

function renderLocalFallback(userText: string, elements: ChatElements): LocalFallbackResult {
  const localReply = window.__nephroLocalChatReply?.(userText);
  if (localReply === null) {
    // The legacy local command handler owns clear semantics and clears its
    // transcript. Keep typed state in sync without persisting that reply.
    resetChat(state);
    setChatLoading(elements, false);
    collapseGuidedPrompts(elements, false);
    elements.messages.parentElement?.querySelector<HTMLElement>('[data-ai-chat-status]')?.remove();
    return 'cleared';
  }
  if (!localReply) return 'none';
  replaceLatestLocalReply(elements, localReply);
  return 'reply';
}



async function submitChat(userText: string, elements: ChatElements): Promise<void> {
  if (state.chatStatus === 'loading') return;
  addChatMessage(state, { role: 'user', content: userText });
  appendChatBubble(elements, 'user', userText);
  collapseGuidedPrompts(elements, true);
  elements.input.value = '';
  state.chatStatus = 'loading';
  elements.send.disabled = true;
  elements.input.setAttribute('aria-busy', 'true');
  setChatLoading(elements, true, 'Checking optional AI availability…');
  setStatus(chatStatusElement(elements), 'loading', 'Preparing a response…');
  try {
    await ensureAiInitialized();
    if (state.aiCapability !== 'enabled') {
      state.chatStatus = state.aiCapability === 'disabled' ? 'disabled' : 'error';
      setChatLoading(elements, false);
      const fallback = renderLocalFallback(userText, elements);
      if (fallback === 'cleared') return;
      setStatus(chatStatusElement(elements), state.chatStatus, fallback === 'reply' ? 'AI assistant is offline. A local guided response was used.' : 'AI assistant is offline. Guided answers remain available.');
      return;
    }
    setChatLoading(elements, true);
    setStatus(chatStatusElement(elements), 'loading', 'AI is analyzing your question…');
    const result = await chat(state.chatMessages, chatContext());
    state.chatStatus = 'ready';
    setChatLoading(elements, false);
    const reply = result.message.trim();
    addChatMessage(state, { role: 'assistant', content: reply });
    replaceLatestLocalReply(elements, reply, 'api');
    setStatus(chatStatusElement(elements), 'ready', 'Response ready. AI attribution is shown beneath the answer.');
    renderAiCapabilityNotice('enabled');
  } catch (error) {
    setChatLoading(elements, false);
    markAiUnavailable(error);
    state.chatStatus = isApiError(error) && error.disabled ? 'disabled' : 'error';
    const fallback = renderLocalFallback(userText, elements);
    if (fallback === 'cleared') return;
    const rateLimited = isApiError(error) && error.status === 429;
    const statusMessage = rateLimited
      ? (fallback === 'reply' ? 'AI is rate limiting requests. A local guided response was used.' : 'AI is rate limiting requests. Try again shortly.')
      : (fallback === 'reply' ? 'AI is unavailable. A local guided response was used.' : 'AI is unavailable. Try a guided starter instead.');
    setStatus(chatStatusElement(elements), state.chatStatus, statusMessage);
  } finally {
    elements.send.disabled = false;
    elements.input.removeAttribute('aria-busy');
  }

}
function connectChat(): void {
  const elements = getChatElements();
  if (!elements) return;
  elements.send.addEventListener('click', () => {
    const value = elements.input.value.trim();
    if (!value) return;
    void submitChat(value, elements);
  }, true);
  elements.quick.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-q]');
    const value = button?.dataset.q || '';
    if (!value) return;
    if (value === 'clear_chat') {
      resetChat(state);
      setChatLoading(elements, false);
      collapseGuidedPrompts(elements, false);
      const status = elements.messages.parentElement?.querySelector<HTMLElement>('[data-ai-chat-status]');
      status?.remove();
      return;
    }
    elements.input.focus();
    // The legacy handler owns predefined prompts so they never call the API.
  }, true);
}

function start(): void {
  window.__nephroTypedChat = true;
  aiInitialization = initializeAi();
  captureSelectedImage();
  connectExploratoryAnalysis();
  observeReports();
  connectChat();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
else start();
