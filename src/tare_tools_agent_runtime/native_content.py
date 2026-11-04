"""Immutable user attachment identities; hosts retain and resolve the bytes.

References carry no filesystem paths, credentials or provider wire objects.
They are part of the user message identity and survive context projection.
"""
from dataclasses import asdict, dataclass, field
import re


MAX_ATTACHMENTS = 16
MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class AttachmentRef:
    name: str
    media_type: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.strip()
                or len(self.name.encode('utf-8')) > 255
                or any(ord(char) < 32 or ord(char) == 127 for char in self.name)):
            raise ValueError('invalid attachment display name')
        if (not isinstance(self.media_type, str) or len(self.media_type) > 127
                or not re.fullmatch(r'[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*', self.media_type)):
            raise ValueError('invalid attachment media type')
        if not isinstance(self.sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', self.sha256):
            raise ValueError('invalid attachment digest')
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_ATTACHMENT_BYTES:
            raise ValueError('invalid attachment byte size')


def validate_attachments(refs):
    if not isinstance(refs, tuple) or not all(isinstance(ref, AttachmentRef) for ref in refs):
        raise TypeError('attachments must be a tuple of AttachmentRef')
    if len(refs) > MAX_ATTACHMENTS or len(set(refs)) != len(refs):
        raise ValueError('too many or duplicate attachments')
    return refs


def attachment_records(refs):
    return [asdict(ref) for ref in validate_attachments(refs)]


def attachments_from_records(records):
    if (not isinstance(records, list) or len(records) > MAX_ATTACHMENTS
            or any(not isinstance(row, dict) for row in records)):
        raise ValueError('invalid attachment records')
    return validate_attachments(tuple(AttachmentRef(**row) for row in records))


def validate_user_text(text, attachments=()):
    validate_attachments(attachments)
    if not isinstance(text, str) or not text.strip() and not attachments:
        raise ValueError('non-empty user text or attachments required')


@dataclass(frozen=True)
class NativeMessage:
    message_id: str
    text: str = field(repr=False)
    attachments: tuple[AttachmentRef, ...] = field(default=(), repr=False)
    interrupts_turn: bool = False
    model_call_limit: int | None = None

    def __post_init__(self):
        if type(self.interrupts_turn) is not bool:
            raise ValueError("invalid native interruption flag")
        if self.model_call_limit is not None and (type(self.model_call_limit) is not int or self.model_call_limit < 1):
            raise ValueError('invalid native model call limit grant')
        if (not isinstance(self.message_id, str) or not self.message_id.strip()
                or len(self.message_id.encode('utf-8')) > 128):
            raise ValueError('invalid native message identity')
        validate_user_text(self.text, self.attachments)
        if len(self.text.encode('utf-8')) > 16384:
            raise ValueError('native continuation input over budget')

    def event(self, after_sequence):
        event = {'kind': 'user_message', 'message_id': self.message_id,
                 'text': self.text, 'after_sequence': after_sequence}
        if self.attachments:
            event['attachments'] = attachment_records(self.attachments)
        if self.interrupts_turn:
            event['interrupts_turn'] = True
        if self.model_call_limit is not None:
            event['model_call_limit'] = self.model_call_limit
        return event

    @classmethod
    def from_event(cls, event):
        keys = {'kind', 'message_id', 'text', 'after_sequence'}
        if 'attachments' in event:
            keys.add('attachments')
            if not event['attachments']:
                raise ValueError('empty attachment field is not canonical')
        if 'interrupts_turn' in event:
            if event['interrupts_turn'] is not True:
                raise ValueError('invalid native interruption flag')
            keys.add('interrupts_turn')
        if 'model_call_limit' in event:
            if event['model_call_limit'] is None:
                raise ValueError('empty native model call limit grant')
            keys.add('model_call_limit')
        if set(event) != keys or event.get('kind') != 'user_message':
            raise ValueError('invalid native message event')
        return cls(event['message_id'], event['text'],
                   attachments_from_records(event.get('attachments', [])), event.get('interrupts_turn', False),
                   event.get('model_call_limit'))
